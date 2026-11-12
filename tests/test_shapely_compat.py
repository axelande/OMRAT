"""``geometries.shapely_compat.transform`` == ``shapely.ops.transform``, no warning.

Newer Shapely deprecates ``shapely.ops.transform``; OMRAT calls go through
the compat helper, which must give the same geometry for array functions,
per-point functions and every geometry type.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import pytest
from shapely import wkt
from shapely.geometry import GeometryCollection, Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.shapely_compat import transform  # noqa: E402

GEOMS = [
    'POINT (11 56)',
    'LINESTRING (11 56, 11.1 56.05, 11.2 56)',
    'POLYGON ((11 56, 11.2 56, 11.2 56.2, 11 56.2, 11 56), (11.05 56.05, 11.1 56.05, 11.1 56.1, 11.05 56.05))',
    'MULTIPOLYGON (((11 56, 11.1 56, 11.1 56.1, 11 56)), ((12 57, 12.1 57, 12.1 57.1, 12 57)))',
    'GEOMETRYCOLLECTION (POINT (11 56), LINESTRING (11 56, 12 57))',
]


def _array_func(x, y):
    """Works on arrays (like pyproj)."""
    return (x - 11.0) * 111_000.0 * math.cos(math.radians(56.0)), (y - 56.0) * 111_000.0


def _per_point_func(x, y):
    """Only works on scalars (like a QgsPointXY or an ``if`` on the value)."""
    if x > 11.05:
        return float(x) * 2.0, float(y) + 1.0
    return float(x), float(y)


def _reference(func, geom):
    from shapely.ops import transform as ops_transform
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ops_transform(func, geom)


@pytest.mark.parametrize('text', GEOMS)
@pytest.mark.parametrize('func', [_array_func, _per_point_func])
def test_same_geometry_as_shapely_ops(text, func):
    geom = wkt.loads(text)
    got = transform(func, geom)
    want = _reference(func, geom)
    assert got.geom_type == want.geom_type
    assert got.equals_exact(want, 1e-6)


def test_no_deprecation_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        transform(_array_func, wkt.loads(GEOMS[2]))
        transform(_per_point_func, wkt.loads(GEOMS[2]))


def test_three_value_func_and_empty_geometries():
    assert transform(lambda x, y: (x + 1, y + 1, 0.0), Point(1, 2)).equals(Point(2, 3))
    assert transform(_array_func, Polygon()).is_empty
    assert transform(_array_func, GeometryCollection()).is_empty
