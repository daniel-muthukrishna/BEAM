"""
Web entry point for tessbeam
"""

import argparse
import atexit
import glob
import os
import shutil
import tempfile
import time as _time

import gradio as gr

from tessbeam import api, sectors, sky

MAX_FRAMES = 350
KEEP_OUTPUTS_S = 3600                         # delete generated files after 1 hour
OUT_DIR = tempfile.mkdtemp(prefix="tessbeam_web_")
atexit.register(shutil.rmtree, OUT_DIR, ignore_errors=True)
INTRO = """
### TESS stray-light predictor (draft)
Enter a time. It decides the TESS sector, and the sector decides where the camera pointing.
The model predicts the Earth/Moon scattered light on the inputted camera as a 256×256 image of its four CCDs
(each pixel = 16×16 real pixels), in e-/s per pixel above the background level. 

Optionally give a target (RA and Dec in degrees). If the given target can be seen at the inputted time,
its place on the detector is marked with a red cross, found with tess-point, and the stray light predicted there is listed.
TESS's position comes from JPL Horizons, which covers 2018-04-18 to 2028-03-29.

A single time takes about 2 s. A GIF takes about 1 s per frame, so a whole sector at the
default 6-hour step (about 110 frames) takes about 2 minutes.

**Camera 3 only for now:** cameras 1, 2 and 4 are not available yet.
"""
# input labels styles
CSS = """
.tb-input span[data-testid="block-info"] {
    font-weight: 700 !important;
    font-size: var(--text-lg) !important;
    color: var(--body-text-color) !important;
}
"""


def _tmp(suffix):
    """New output path in OUT_DIR; prunes files older than KEEP_OUTPUTS_S first."""
    cutoff = _time.time() - KEEP_OUTPUTS_S
    for f in glob.glob(os.path.join(OUT_DIR, "*")):
        try:
            if os.path.getmtime(f) < cutoff:
                os.remove(f)
        except OSError:
            pass
    fd, path = tempfile.mkstemp(suffix=suffix, dir=OUT_DIR)
    os.close(fd)
    return path


def _check_inputs(**values):
    missing = [k for k, v in values.items() if v is None or (isinstance(v, str) and not v.strip())]
    if missing:
        raise gr.Error(f"please fill in: {', '.join(missing)}", duration=None)


def _user_error(e):
    """Show the reason in the page (until dismissed) instead of Gradio's bare 'Error'."""
    plain = isinstance(e, (ValueError, NotImplementedError, ImportError))      # a bad request
    return gr.Error(str(e) if plain else f"{type(e).__name__}: {e}", duration=None)


def _target(target_ra, target_dec):
    """The two optional text boxes -> (RA, Dec) in degrees, or None when both are empty."""
    ra, dec = ("" if v is None else str(v).strip() for v in (target_ra, target_dec))
    if not ra and not dec:
        return None
    if not ra or not dec:
        raise gr.Error("give both Target RA and Target Dec, or leave both empty", duration=None)
    try:
        return (float(ra), float(dec))
    except ValueError:
        raise gr.Error(f"Target RA and Dec must be numbers in degrees, got {ra!r} and {dec!r}", duration=None)


def _note(s, series=False):
    """What was predicted, in words: sector and pointing, target, and anything to be careful about.
    (The training-range check, Series.out_of_range, is kept for development but not shown.)"""
    n = len(s.tjd)
    lines = [f"**Sector {s.sector}**: spacecraft pointing RA {s.ra:.4f}°, Dec {s.dec:.4f}°, "
             f"roll {s.roll:.4f}° (sector table built {sectors.built()[:10]})."]
    if s.planned:
        lines.append("⚠️ This sector is **planned**, not flown yet: its pointing is provisional and "
                     "its dates are good to about a day.")
    if s.target is not None:
        t, f = s.target, s.target_flux
        value = f"{f[0]:.1f}" if n == 1 else f"{f.min():.1f} to {f.max():.1f}"
        lines.append(f"**Target** RA {t.ra:.4f}°, Dec {t.dec:.4f}° is on camera {t.camera}, CCD {t.ccd}, "
                     f"column {t.col:.1f}, row {t.row:.1f} (full-frame image pixel, as tess-point counts "
                     f"it). Predicted stray light there: {value} e-/s per pixel.")
    elif series:
        lines.append("Give a target RA and Dec to see a zoom around it and the stray light there as a light curve.")
    else:
        lines.append("Give a target RA and Dec to see a zoom around it.")
    if s.skipped:
        lines.append(f"Skipped {len(s.skipped)} time(s) outside JPL's TESS ephemeris, "
                     f"{s.skipped[0]} to {s.skipped[-1]} UTC.")
    return "\n\n".join(lines)


def _rows(s):
    """One table row per frame; the last column is the stray light at the target (blank without one)."""
    at_target = [""] * len(s.tjd) if s.target is None else [round(float(v), 1) for v in s.target_flux]
    return [[u, round(float(ed), 1), round(float(e3), 1), round(float(m3), 1), v]
            for u, ed, e3, m3, v in zip(s.times_utc, s.angles["ED"], s.angles["E3el"], s.angles["M3el"], at_target)]


def run_single(time, target_ra, target_dec, seed):
    _check_inputs(Time=time, Seed=seed)
    try:
        p = api.predict(time, target=_target(target_ra, target_dec), camera=3, seed=int(seed))
        rows = [[k, round(v, 2)] for k, v in p.angles.items()]
        zoom = None if p.target is None else p.to_zoom_png(_tmp(".png"))
        return p.to_png(_tmp(".png")), zoom, _note(p._series()), p.to_fits(_tmp(".fits")), rows
    except gr.Error:
        raise
    except Exception as e:
        raise _user_error(e)


def _series_outputs(s):
    zoom = None if s.target is None else s.to_zoom_gif(_tmp(".gif"))
    curve = None if s.target is None else s.to_lightcurve(_tmp(".png"))
    return s.to_gif(_tmp(".gif")), zoom, curve, _note(s, series=True), s.to_fits(_tmp(".fits")), _rows(s)


def run_series(start, end, step, target_ra, target_dec, seed, progress=gr.Progress()):
    _check_inputs(Start=start, End=end, Step=step, Seed=seed)
    try:
        return _series_outputs(api.predict_series(
            start, end, step, target=_target(target_ra, target_dec), camera=3, seed=int(seed),
            max_frames=MAX_FRAMES, progress=lambda i, n: progress((i, n), desc="generating frames")))
    except gr.Error:
        raise
    except Exception as e:
        raise _user_error(e)


def run_sector(sector, step, target_ra, target_dec, seed, progress=gr.Progress()):
    _check_inputs(Sector=sector, Step=step, Seed=seed)
    try:
        return _series_outputs(api.predict_sector(
            sector, step, target=_target(target_ra, target_dec), camera=3, seed=int(seed),
            max_frames=MAX_FRAMES, progress=lambda i, n: progress((i, n), desc="generating frames")))
    except gr.Error:
        raise
    except Exception as e:
        raise _user_error(e)


def _gif_outputs():
    """The output widgets of a GIF tab (shared by the range and whole-sector tabs): the GIF and the
    zoom around the target side by side, then the light curve beside the note and the download, the
    table folded away."""
    with gr.Row():
        gif = gr.Image(type="filepath", label="Prediction (GIF)")
        zoom = gr.Image(type="filepath", label="Zoom around the target (GIF)")
    with gr.Row():
        curve = gr.Image(type="filepath", label="Light curve at the target")
        with gr.Column():
            note = gr.Markdown()
            fits = gr.File(label="FITS cube")
    with gr.Accordion("Per-frame values", open=False):
        table = gr.Dataframe(headers=["UTC", "ED", "E3el", "M3el", "at target (e-/s)"], label="Per frame")
    return [gif, zoom, curve, note, fits, table]

def build():
    # delete_cache: Gradio's copies of returned files are removed hourly after an hour
    with gr.Blocks(title="TESS stray light (draft)", css=CSS,
                   delete_cache=(KEEP_OUTPUTS_S, KEEP_OUTPUTS_S)) as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            # text boxes, not gr.Number: an empty number box shows (and sends) 0, and RA 0, Dec 0
            # is a real place on the sky, so "no target" could not be told apart from a target
            target_ra = gr.Textbox(value="", label="Target RA (deg)", placeholder="optional", elem_classes="tb-input")
            target_dec = gr.Textbox(value="", label="Target Dec (deg)", placeholder="optional", elem_classes="tb-input")
            seed = gr.Number(value=0, precision=0, label="Seed", elem_classes="tb-input")
        # one shared queue slot: the three buttons take turns on the GPU
        turn = dict(concurrency_id="gpu", concurrency_limit=1)
        with gr.Tab("Single time"):
            time = gr.Textbox(value="2019-05-09T05:15:00", label="Time (UTC ISO, or TJD)", elem_classes="tb-input")
            go1 = gr.Button("Predict", variant="primary")
            with gr.Row():
                img = gr.Image(type="filepath", label="Prediction")
                zoom1 = gr.Image(type="filepath", label="Zoom around the target")
            note1 = gr.Markdown()
            fits1 = gr.File(label="FITS")
            with gr.Accordion("Geometry of this frame", open=False):
                tab1 = gr.Dataframe(headers=["value", "deg / Earth radii"], label="Geometry")
            go1.click(run_single, [time, target_ra, target_dec, seed], [img, zoom1, note1, fits1, tab1], **turn)
        with gr.Tab("Time range → GIF"):
            with gr.Row():
                start = gr.Textbox(value="2019-05-08T05:00:00", label="Start", elem_classes="tb-input")
                end = gr.Textbox(value="2019-05-10T05:00:00", label="End", elem_classes="tb-input")
                step = gr.Textbox(value="1h", label=f"Step (e.g. 30min, 2h; max {MAX_FRAMES} frames)", elem_classes="tb-input")
            go2 = gr.Button("Make GIF", variant="primary")
            go2.click(run_series, [start, end, step, target_ra, target_dec, seed], _gif_outputs(), **turn)
        with gr.Tab("Whole sector → GIF"):
            with gr.Row():
                sector = gr.Number(value=11, precision=0, label="Sector", elem_classes="tb-input")
                step3 = gr.Textbox(value="6h", label=f"Step (max {MAX_FRAMES} frames)", elem_classes="tb-input")
            go3 = gr.Button("Make GIF of the whole sector", variant="primary")
            go3.click(run_sector, [sector, step3, target_ra, target_dec, seed], _gif_outputs(), **turn)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print("loading the sector table, checking JPL Horizons and loading the model ...", flush=True)
    numbers = sectors.numbers()               # refreshes the table first if it is more than a week old
    print(f"sector table built {sectors.built()[:10]}: {len(numbers)} sectors", flush=True)
    try:
        import tess_stars2px  # noqa: F401
    except ImportError:
        print("tess-point is not installed: targets cannot be marked (pip install tess-point)", flush=True)
    else:
        # every sector's RA/Dec grid now, from the stored WCS fits, so that no request waits for one
        t = _time.time()
        print(f"RA/Dec grids ready for {sky.precompute(numbers)} sectors ({_time.time() - t:.0f} s)", flush=True)
    # one real position lookup: fails at startup (not on the first click) if Horizons
    # can't be reached and nothing is cached yet
    api.compute_geometry(*sectors.pointing(11), "2019-05-09T05:15:00")
    api.get_model()
    print(f"ready: http://{args.host}:{args.port}", flush=True)
    build().queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
