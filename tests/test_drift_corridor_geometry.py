"""Unit tests for compute/drift_corridor_geometry.py.

Covers the four small helpers: _compass_idx_to_math_idx,
_extract_obstacle_segments, _create_drift_corridor,
_segment_intersects_corridor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import LinearRing

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.drift_corridor_geometry import (
    _compass_idx_to_math_idx,
    _extract_obstacle_segments,
    _create_drift_corridor,
    _segment_intersects_corridor,
)


# ---------------------------------------------------------------------------
# _compass_idx_to_math_idx
# ---------------------------------------------------------------------------

class TestCompassIdxToMathIdx:
    @pytest.mark.parametrize("compass_idx, math_idx", [
        (0, 2),  # compass=0 (N), math_angle=90 (N in math), math_idx=2
        (1, 1),  # compass=45 (NE), math_angle=45, math_idx=1
        (2, 0),  # compass=90 (E), math_angle=0, math_idx=0
        (3, 7),  # compass=135 (SE), math_angle=315, math_idx=7
        (4, 6),  # compass=180 (S), math_angle=270, math_idx=6
        (5, 5),  # compass=225 (SW), math_angle=225, math_idx=5
        (6, 4),  # compass=270 (W), math_angle=180, math_idx=4
        (7, 3),  # compass=315 (NW), math_angle=135, math_idx=3
    ])
    def test_all_cardinals_and_ordinals(self, compass_idx, math_idx):
        assert _compass_idx_to_math_idx(compass_idx) == math_idx


# ---------------------------------------------------------------------------
# _extract_obstacle_segments
# ---------------------------------------------------------------------------

class TestExtractObstacleSegments:
    def test_polygon_produces_n_segments(self):
        poly = box(0, 0, 10, 10)
        segs = _extract_obstacle_segments(poly)
        # A box has 4 edges.
        assert len(segs) == 4

    def test_multipolygon_aggregates_segments(self):
        mp = MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)])
        segs = _extract_obstacle_segments(mp)
        assert len(segs) == 8

    def test_polygon_with_hole_includes_inner_ring(self):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(outer, [hole])
        segs = _extract_obstacle_segments(poly)
        # 4 outer + 4 inner = 8
        assert len(segs) == 8

    def test_empty_polygon_yields_no_segments(self):
        assert _extract_obstacle_segments(Polygon()) == []

    def test_segments_close_the_ring(self):
        """The last segment should close back to the first vertex."""
        poly = box(0, 0, 1, 1)
        segs = _extract_obstacle_segments(poly)
        # Each segment is ((x0, y0), (x1, y1)); the end of the last
        # segment must equal the start of the first.
        assert segs[-1][1] == segs[0][0]

    def test_linestring_not_an_obstacle(self):
        """LineString is not Polygon/MultiPolygon -> no segments extracted."""
        ls = LineString([(0, 0), (10, 0), (10, 10)])
        assert _extract_obstacle_segments(ls) == []

    def test_point_not_an_obstacle(self):
        """Point is not Polygon/MultiPolygon -> no segments extracted."""
        assert _extract_obstacle_segments(Point(0, 0)) == []


# ---------------------------------------------------------------------------
# _create_drift_corridor
# ---------------------------------------------------------------------------

class TestCreateDriftCorridor:
    def test_simple_east_leg_north_drift(self):
        leg = LineString([(0, 0), (1000, 0)])
        c = _create_drift_corridor(leg, drift_angle=90.0, distance=500.0,
                                   lateral_spread=50.0)
        assert isinstance(c, Polygon) and c.area > 100 * 1000
        # Corridor extends north of the leg.
        assert c.bounds[3] > 400

    def test_zero_length_leg_returns_none(self):
        leg = LineString([(0, 0), (0, 0)])
        assert _create_drift_corridor(leg, 0.0, 100.0, 50.0) is None

    def test_single_point_leg_returns_none(self):
        # LineString requires >= 2 points; mimic a degenerate pair.
        leg = LineString([(5, 5), (5, 5)])
        assert _create_drift_corridor(leg, 0.0, 100.0, 50.0) is None

    def test_zero_distance_and_spread_returns_none(self):
        """Zero distance + zero spread -> degenerate corridor with zero
        area -> function returns None via the ``corridor.area == 0`` guard."""
        leg = LineString([(0, 0), (1000, 0)])
        assert _create_drift_corridor(leg, 90.0, distance=0.0, lateral_spread=0.0) is None

    def test_empty_linestring_returns_none(self):
        """A LINESTRING EMPTY has no coords -> ``len(coords) < 2`` guard."""
        leg = LineString()
        assert _create_drift_corridor(leg, 90.0, 100.0, 50.0) is None


# ---------------------------------------------------------------------------
# _segment_intersects_corridor
# ---------------------------------------------------------------------------

class TestSegmentIntersectsCorridor:
    def test_segment_inside_corridor_returns_true(self):
        corridor = box(0, 0, 1000, 1000)
        seg = ((100, 100), (200, 200))
        assert _segment_intersects_corridor(seg, corridor)

    def test_segment_outside_corridor_returns_false(self):
        corridor = box(0, 0, 100, 100)
        seg = ((500, 500), (600, 600))
        assert not _segment_intersects_corridor(seg, corridor)

    def test_segment_crossing_corridor_boundary_returns_true(self):
        """A segment that crosses the corridor (not just touching a
        vertex) counts as a substantial intersection."""
        corridor = box(0, 0, 100, 100)
        seg = ((50, -10), (50, 110))  # runs through the corridor
        assert _segment_intersects_corridor(seg, corridor)

    def test_extra_args_do_not_crash(self):
        """Passing the optional drift_angle / leg arguments doesn't
        change the behaviour for an inside-the-corridor segment."""
        corridor = box(0, 0, 1000, 1000)
        seg = ((500, 500), (600, 600))  # well inside, easy hit
        leg_line = LineString([(0, 0), (1000, 0)])
        # Just verify the call path runs without exceptions.
        result = _segment_intersects_corridor(
            seg, corridor, drift_angle=90.0,
            leg_centroid=(500, 0), leg_line=leg_line,
        )
        assert isinstance(result, bool)

    def test_zero_length_segment_with_direction_returns_false(self):
        """Degenerate segment (p1==p2) with drift_angle+leg_centroid set
        hits the ``seg_len == 0`` guard and returns False."""
        corridor = box(0, 0, 1000, 1000)
        seg = ((500, 500), (500, 500))
        result = _segment_intersects_corridor(
            seg, corridor, drift_angle=90.0, leg_centroid=(500, 0),
        )
        assert result is False

    def test_point_touch_only_not_substantial(self):
        """A segment that only touches the corridor at a single vertex
        (and has no interior or midpoint inside) returns False via the
        Point-touch guard."""
        corridor = box(0, 0, 100, 100)
        # Segment that grazes the corner (100, 100) -- touches at one point only.
        seg = ((100, 100), (200, 200))
        # All of interior/mid/opposite are outside the corridor.
        assert _segment_intersects_corridor(seg, corridor) is False
