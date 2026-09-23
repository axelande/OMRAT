"""Canonical boundary-edge ordering (geometries/boundary_edges.py).

The powered ray caster and the result layers must number an obstacle's
edges identically; these tests pin the order so ``segment_idx`` in the
layer and ``seg_<idx>`` in ``by_obstacle_segment_legdir`` stay in step.
Pure shapely -- runs with ``--noconftest -p no:qgis``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.boundary_edges import boundary_edges  # noqa: E402


def _signed_area(edges) -> float:
    return 0.5 * sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in edges)


class TestBoundaryEdges:
    def test_box_gives_four_edges_ccw(self):
        edges = boundary_edges(box(0, 0, 2, 1))
        assert len(edges) == 4
        assert _signed_area(edges) > 0          # counter-clockwise

    def test_clockwise_input_is_reoriented_to_same_order(self):
        ccw = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
        cw = Polygon([(0, 0), (0, 1), (2, 1), (2, 0)])
        assert boundary_edges(cw) == boundary_edges(ccw)

    def test_zero_length_edges_dropped(self):
        poly = Polygon([(0, 0), (2, 0), (2, 0), (2, 1), (0, 1), (0, 1)])
        edges = boundary_edges(poly)
        assert len(edges) == 4
        assert all((x1, y1) != (x2, y2) for x1, y1, x2, y2 in edges)

    def test_hole_edges_follow_exterior_and_run_clockwise(self):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(4, 4), (6, 4), (6, 6), (4, 6)]          # given CCW on purpose
        edges = boundary_edges(Polygon(outer, [hole]))
        assert len(edges) == 8
        assert _signed_area(edges[:4]) > 0
        assert _signed_area(edges[4:]) < 0              # hole re-oriented CW

    def test_multipolygon_concatenates_in_member_order(self):
        a, b = box(0, 0, 1, 1), box(5, 0, 6, 1)
        edges = boundary_edges(MultiPolygon([a, b]))
        assert edges == boundary_edges(a) + boundary_edges(b)

    def test_linestring_follows_vertex_path(self):
        edges = boundary_edges(LineString([(0, 0), (1, 0), (1, 1)]))
        assert edges == [(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 1.0, 1.0)]

    def test_point_and_empty_yield_nothing(self):
        assert boundary_edges(Point(1, 1)) == []
        assert boundary_edges(Polygon()) == []
        assert boundary_edges(None) == []
