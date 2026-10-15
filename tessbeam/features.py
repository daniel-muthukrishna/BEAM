"""
Geometry values -> the model's 16 standardized conditioning features
"""

import json
import os
import warnings

import numpy as np

CONTEXT_FEATURE_NAMES = [
    '1/ED', '1/MD', '1/ED^2', '1/MD^2',
    'Eel', 'sin(Eaz)', 'cos(Eaz)',
    'Mel', 'sin(Maz)', 'cos(Maz)',
    'Ecam_el', 'sin(Ecam_az)', 'cos(Ecam_az)',
    'Mcam_el', 'sin(Mcam_az)', 'cos(Mcam_az)',
]
TRAIN_RANGES = os.path.join(os.path.dirname(__file__), "data", "train_ranges.json")


def training_angles(g, camera):
    """Geometry dict from tessbeam.geometry.emi -> the angle dict the training code used."""
    c = str(camera)
    r = {k: np.round(np.asarray(v, dtype=np.float64), 1) for k, v in g.items()}
    rad = lambda x: np.round(np.radians(x), 5)
    return {
        '1/ED': np.round(1 / (r['ED'] / 50), 5),
        '1/MD': np.round(1 / (r['MD'] / 50), 5),
        '1/ED^2': np.round(1 / (r['ED'] / 50) ** 2, 5),
        '1/MD^2': np.round(1 / (r['MD'] / 50) ** 2, 5),
        'Eel': rad(r['Eel']), 'Eaz': rad(r['Eaz']),
        'Mel': rad(r['Mel']), 'Maz': rad(r['Maz']),
        f'E{c}el': rad(r[f'E{c}el']), f'E{c}az': rad(r[f'E{c}az']),
        f'M{c}el': rad(r[f'M{c}el']), f'M{c}az': rad(r[f'M{c}az']),
    }


def build_context_features(a, camera):
    """Vectorized copy of beam.data.datasets.build_context_features -> (N, 16)."""
    c = str(camera)
    return np.stack([
        a['1/ED'], a['1/MD'],
        a['1/ED^2'], a['1/MD^2'],
        a['Eel'], np.sin(a['Eaz']), np.cos(a['Eaz']),
        a['Mel'], np.sin(a['Maz']), np.cos(a['Maz']),
        a['E' + c + 'el'], np.sin(a['E' + c + 'az']), np.cos(a['E' + c + 'az']),
        a['M' + c + 'el'], np.sin(a['M' + c + 'az']), np.cos(a['M' + c + 'az']),
    ], axis=-1).astype(np.float64).reshape(-1, 16)


def context_vectors(g, camera, context_stats):
    """Geometry dict -> standardized (N, 16) float32 features for the model."""
    names = list(context_stats.get('feature_names', CONTEXT_FEATURE_NAMES))
    if names != CONTEXT_FEATURE_NAMES:
        raise ValueError(f"model expects features {names}, this code builds {CONTEXT_FEATURE_NAMES}")
    if context_stats.get('add_exposure_flag', False):
        raise NotImplementedError("models trained with add_exposure_flag=True are not supported yet")
    x = build_context_features(training_angles(g, camera), camera)
    mean = np.asarray(context_stats['mean'], dtype=np.float64)
    std = np.asarray(context_stats['std'], dtype=np.float64)
    return ((x - mean) / std).astype(np.float32)


def out_of_range(ctx, model_dir=None, camera=None, path=TRAIN_RANGES):
    """For each frame, the names of features outside the training min/max (empty = in range).

    The ranges are only used if they were built for this model directory and camera;
    otherwise no frame is flagged (with a warning).
    """
    empty = [[] for _ in range(len(ctx))]
    if not os.path.exists(path):
        return empty
    r = json.load(open(path))
    same_model = model_dir is None or os.path.realpath(r.get('model_dir', '')) == os.path.realpath(model_dir)
    if not same_model or (camera is not None and int(r.get('camera', -1)) != int(camera)):
        warnings.warn(f"{path} was built for {r.get('model_dir')} camera {r.get('camera')}; "
                      "skipping the training-range check")
        return empty
    # compare in float32 (the precision of ctx) with a small tolerance, so frames sitting
    # exactly on a training extreme are not flagged
    lo = np.asarray(r['min'], dtype=np.float32) - 1e-5
    hi = np.asarray(r['max'], dtype=np.float32) + 1e-5
    bad = (ctx < lo) | (ctx > hi)
    return [[CONTEXT_FEATURE_NAMES[j] for j in np.where(row)[0]] for row in bad]
