"""``shapely.ops.transform`` without its deprecation warning (pure; no QGIS).

Newer Shapely deprecates ``shapely.ops.transform(func, geom)`` in favour of
the array-based ``shapely.transform``.  :func:`transform` keeps the old call
shape -- ``func(x, y)`` gets the x and y coordinates as arrays and returns
the new ones, or, when it only handles one point at a time (it raises on
arrays, e.g. a ``QgsPointXY`` or an ``if x > 0``), it is called per point --
and runs on ``shapely.transform``.  Shapely 1.8 has no ``shapely.transform``
and keeps ``shapely.ops.transform``.

Use this instead of ``shapely.ops.transform`` everywhere in OMRAT.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def transform(func: Callable[..., Any], geom: Any) -> Any:
    """``geom`` with every (x, y) replaced by ``func(x, y)`` (z dropped)."""
    try:
        import shapely
        array_transform = shapely.transform
    except AttributeError:  # Shapely 1.8
        from shapely.ops import transform as ops_transform
        return ops_transform(func, geom)

    def _apply(coords: np.ndarray) -> np.ndarray:
        n = len(coords)
        if n == 0:
            return coords
        try:
            res = func(coords[:, 0], coords[:, 1])
            out = np.column_stack([
                np.broadcast_to(np.asarray(res[0], dtype=float), (n,)),
                np.broadcast_to(np.asarray(res[1], dtype=float), (n,)),
            ])
        except Exception:  # nosec B110 - a per-point func: call it point by point
            out = np.array([tuple(func(float(x), float(y)))[:2] for x, y in coords], dtype=float)
        return out

    return array_transform(geom, _apply)
