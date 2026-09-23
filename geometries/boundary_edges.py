"""Canonical boundary-edge ordering shared by compute and the result layers.

The powered ray caster (``geometries/get_powered_overlap.py``) reports which
boundary edge of an obstacle each ray hits, and the result layer
(``geometries/result_layers.py``) draws one feature per boundary edge.  Both
must slice a geometry into edges in *exactly* the same order for the
``segment_idx`` of a feature to mean the same edge as the ``seg_<idx>`` key
in ``by_obstacle_segment_legdir``.  This module is the single definition of
that order:

* Polygons are oriented counter-clockwise (holes clockwise) first, so the
  right-hand perpendicular of every edge is the outward normal.
* Exterior ring edges come first, then each interior ring in order.
* A MultiPolygon yields each member polygon's edges in sequence.
* Zero-length edges (repeated vertices) are dropped.

No QGIS dependency -- shapely only.
"""

from __future__ import annotations

from typing import Iterable

from shapely.geometry import LinearRing, LineString, MultiPolygon, Polygon
from shapely.geometry import polygon as shapely_polygon

Edge = tuple[float, float, float, float]


def _ring_edges(coords: Iterable) -> list[Edge]:
    pts = [tuple(c[:2]) for c in coords]
    edges: list[Edge] = []
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if (x1, y1) != (x2, y2):
            edges.append((float(x1), float(y1), float(x2), float(y2)))
    return edges


def boundary_edges(geom) -> list[Edge]:
    """Return ``[(x1, y1, x2, y2), ...]`` for every boundary edge of *geom*.

    The list index is the canonical ``segment_idx`` of that edge.  Points and
    empty geometries yield an empty list.
    """
    if geom is None or getattr(geom, 'is_empty', True):
        return []
    if geom.geom_type in ('Point', 'MultiPoint'):
        return []                                   # measure-zero: a ray cannot hit it
    edges: list[Edge] = []
    if isinstance(geom, Polygon):
        oriented = shapely_polygon.orient(geom, sign=1.0)
        edges.extend(_ring_edges(oriented.exterior.coords))
        for interior in oriented.interiors:
            edges.extend(_ring_edges(interior.coords))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            edges.extend(boundary_edges(poly))
    elif isinstance(geom, (LineString, LinearRing)):
        edges.extend(_ring_edges(geom.coords))
    elif hasattr(geom, 'geoms'):
        # MultiLineString / GeometryCollection: members in order.
        for sub in geom.geoms:
            edges.extend(boundary_edges(sub))
    elif hasattr(geom, 'boundary'):
        boundary = geom.boundary
        if hasattr(boundary, 'coords'):
            edges.extend(_ring_edges(boundary.coords))
        elif hasattr(boundary, 'geoms'):
            for line in boundary.geoms:
                edges.extend(_ring_edges(line.coords))
    return edges
