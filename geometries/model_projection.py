"""Local projections (lon/lat -> metres) and bearings measured in them.

Pure numpy, no QGIS / pyproj / matplotlib, so the compute modules can use
it.  :class:`ModelProjector` (via :func:`model_projector`) is the model's
projection: the powered models cast their rays in it and the ship-ship
collision model takes the leg bearings at the waypoints from it.

:func:`projected_bearing` replaces the planar ``atan2(dlon, dlat)`` the
collision model and the junction defaults used before 2026-11-12: in raw
degrees a degree of longitude counts as much as a degree of latitude,
although at 57 N it is only 0.54 of one, so a 44 degree turn came out as
28 and a 50 degree one as 80 (KNOWN_ISSUES 36).
"""

from __future__ import annotations

import math
from math import cos, radians, sin, sqrt
from typing import Any

import numpy as np


class SimpleProjector:
    """Equirectangular projection (lon/lat -> metres) around a reference point."""

    def __init__(self, lon_ref: float, lat_ref: float) -> None:
        self.lon_ref = lon_ref
        self.lat_ref = lat_ref
        self.mx = 111_320.0 * cos(radians(lat_ref))
        self.my = 110_540.0

    def transform(self, lon: float, lat: float) -> tuple[float, float]:
        return ((lon - self.lon_ref) * self.mx,
                (lat - self.lat_ref) * self.my)


class ModelProjector:
    """Local projection (lon/lat -> metres) centred on the model.

    Spherical transverse Mercator about the central meridian ``lon_ref``,
    scaled with the WGS84 radii of curvature at ``lat_ref`` -- the prime
    vertical ``N0`` across, the meridian ``M0`` along -- so the scale is
    right in both directions at the centre.  Away from it the error is the
    Mercator term ``x**2 / (2 R**2)`` (0.006 % at 70 km from the central
    meridian) plus the change of the ellipsoid's curvature with latitude
    (under 0.02 % per degree): distances across a sea area like the
    Kattegat are right within 0.05 %, angles within 0.1 degree.  Pure numpy
    (no pyproj); ``transform`` takes scalars or arrays like
    :class:`SimpleProjector`.

    Replaces the equirectangular :class:`SimpleProjector` around the first
    leg's start point, whose single ``cos(lat0)`` made east-west distances
    3.3 % short 1.2 degrees away and the result depend on which leg is
    first in the route table (KNOWN_ISSUES 27).
    """

    _A = 6378137.0
    _E2 = 6.69437999014e-3

    def __init__(self, lon_ref: float, lat_ref: float) -> None:
        self.lon_ref = float(lon_ref)
        self.lat_ref = float(lat_ref)
        phi0 = radians(self.lat_ref)
        w = 1.0 - self._E2 * sin(phi0) ** 2
        self.n0 = self._A / sqrt(w)                       # prime vertical radius
        self.m0 = self._A * (1.0 - self._E2) / w ** 1.5   # meridian radius
        self._lam0 = radians(self.lon_ref)
        self._phi0 = phi0

    def transform(self, lon, lat):
        lam = np.radians(lon) - self._lam0
        phi = np.radians(lat)
        b = np.cos(phi) * np.sin(lam)
        x = self.n0 * np.arctanh(b)
        y = self.m0 * (np.arctan2(np.tan(phi), np.cos(lam)) - self._phi0)
        return x, y


def _parse_point(coord_str: Any) -> tuple[float, float]:
    parts = str(coord_str).strip().split()
    return float(parts[0]), float(parts[1])


def model_projector(segment_data: dict) -> ModelProjector:
    """:class:`ModelProjector` centred on the bounding box of every leg end point.

    Raises ``ValueError`` when no leg has a readable point.
    """
    lons: list[float] = []
    lats: list[float] = []
    for seg in (segment_data or {}).values():
        if not isinstance(seg, dict):
            continue
        for key in ("Start_Point", "End_Point"):
            try:
                lon, lat = _parse_point(seg[key])
            except Exception:  # nosec B110 B112 - a leg without points does not move the centre
                continue
            lons.append(lon)
            lats.append(lat)
    if not lons:
        raise ValueError("no leg end points to centre the projection on")
    return ModelProjector((min(lons) + max(lons)) / 2.0, (min(lats) + max(lats)) / 2.0)


def model_projector_or_none(segment_data: dict) -> ModelProjector | None:
    """:func:`model_projector`, or ``None`` when no leg has a readable point."""
    try:
        return model_projector(segment_data)
    except ValueError:
        return None


def projected_bearing(
    start: tuple[float, float],
    end: tuple[float, float],
    projector: ModelProjector | SimpleProjector | None = None,
) -> float:
    """Compass bearing (degrees, 0 = north, clockwise) from ``start`` to ``end``.

    Measured between the two points projected with ``projector`` -- pass
    the model's (:func:`model_projector`) so every bearing of a run lies
    in one plane.  ``None`` centres a :class:`ModelProjector` on ``start``,
    which gives the true bearing there.  The projection is conformal, so
    the angle between two bearings taken at the same point does not depend
    on where it is centred (grid north and true north differ by the same
    amount for both).  Coincident points give 0.
    """
    proj = projector if projector is not None else ModelProjector(start[0], start[1])
    x0, y0 = proj.transform(start[0], start[1])
    x1, y1 = proj.transform(end[0], end[1])
    return math.degrees(math.atan2(float(x1) - float(x0), float(y1) - float(y0))) % 360.0
