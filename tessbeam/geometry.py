"""
Spacecraft pointing (RA, Dec, roll) + TESS/Moon positions -> the 22 Earth/Moon
geometry values the model was trained on (the POC "EMI" quantities).

Conventions :
  body frame   +Z = (RA, Dec); +X at position angle -roll (north through east);
               Y = Z x X. Same as tess-point's sky_to_sc_mat / the FFI quaternion.
  camera c     body frame rotated about +Y by CAM_TILT_DEG[c].
  ED, MD       distance TESS->Earth / TESS->Moon in units of RE_KM.
  Eel, Eaz     elevation above the body XY plane, azimuth atan2(y, x)  (degrees).
  E{c}el       angle between camera c's boresight and the body (0-180 deg);
               despite the name it is not an elevation.
  E{c}az       atan2(y_c, x_c) in the camera frame, mod 360; +180 for cameras 1-2.
  Moon values (M...) are defined the same way.
"""

import numpy as np

RE_KM = 6394.573129            # the POC's "Earth radius" used for ED/MD
CAM_TILT_DEG = {1: -36.0, 2: -12.0, 3: 12.0, 4: 36.0}
CAMERAS = (1, 2, 3, 4)


def body_matrix(ra, dec, roll):
    """3x3 matrix whose rows are the spacecraft X, Y, Z axes in ICRS.

    ra, dec: boresight (deg). roll: deg, tess-point convention (e.g. sector 11
    = 138.0761; FFI header SC_ROLL values are equal mod 360).
    Body-frame vector = body_matrix(...) @ icrs_vector.
    """
    a, d = np.radians(ra), np.radians(dec)
    z = np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    north = np.array([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)])
    east = np.array([-np.sin(a), np.cos(a), 0.0])
    pa = np.radians(-roll)
    x = np.cos(pa) * north + np.sin(pa) * east
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def _body_angles(v_body):
    """v_body: (N, 3) -> distance (km), elevation, azimuth (deg) about body +Z."""
    dist = np.linalg.norm(v_body, axis=1)
    u = v_body / dist[:, None]
    el = np.degrees(np.arcsin(np.clip(u[:, 2], -1.0, 1.0)))
    az = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
    return dist, el, az, u


def _camera_angles(u_body, camera):
    """Unit body vectors (N, 3) -> (boresight angle, azimuth) in degrees for camera c."""
    t = np.radians(CAM_TILT_DEG[camera])
    zc = np.array([np.sin(t), 0.0, np.cos(t)])
    xc = np.array([np.cos(t), 0.0, -np.sin(t)])
    ang = np.degrees(np.arccos(np.clip(u_body @ zc, -1.0, 1.0)))
    az = np.degrees(np.arctan2(u_body[:, 1], u_body @ xc)) % 360.0
    if camera in (1, 2):
        az = (az + 180.0) % 360.0
    return ang, az


def emi(ra, dec, roll, tess_geo_km, moon_geo_km):
    """
    Geometry values for every time.

    tess_geo_km, moon_geo_km: (N, 3) geocentric ICRS positions (km) of TESS and the Moon.
    Returns a dict of (N,) arrays (degrees / Earth radii):
      ED, MD, Eel, Eaz, Mel, Maz, and E{c}el, E{c}az, M{c}el, M{c}az for c = 1..4.
    """
    tess = np.atleast_2d(np.asarray(tess_geo_km, dtype=np.float64))
    moon = np.atleast_2d(np.asarray(moon_geo_km, dtype=np.float64))
    m = body_matrix(ra, dec, roll)
    earth_b = (-tess) @ m.T                 # Earth as seen from TESS, body frame
    moon_b = (moon - tess) @ m.T            # Moon as seen from TESS, body frame

    out = {}
    for tag, v in (("E", earth_b), ("M", moon_b)):
        dist, el, az, u = _body_angles(v)
        out[f"{tag}D"] = dist / RE_KM
        out[f"{tag}el"] = el
        out[f"{tag}az"] = az
        for c in CAMERAS:
            out[f"{tag}{c}el"], out[f"{tag}{c}az"] = _camera_angles(u, c)
    return out
