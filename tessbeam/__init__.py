"""
tessbeam: predict TESS stray light for a time (the time decides the sector, and the
sector decides the spacecraft pointing).

    from tessbeam import predict, predict_series, predict_sector
    p = predict("2019-05-09T05:15:00")                       # one frame
    p.to_png("frame.png")
    s = predict_sector(11, target=(193.3, -62.1))            # a whole sector, target marked
    s.to_gif("sector11.gif")

The geometry/feature code (tessbeam.geometry, tessbeam.features, tessbeam.ephem,
tessbeam.sectors) does not import torch; the model is loaded lazily on the first prediction.
"""

__version__ = "0.0.2"


def __getattr__(name):
    # lazy re-exports so `import tessbeam` stays light
    if name in ("predict", "predict_series", "predict_sector", "compute_geometry", "Prediction", "Series"):
        from tessbeam import api
        return getattr(api, name)
    raise AttributeError(f"module 'tessbeam' has no attribute {name!r}")
