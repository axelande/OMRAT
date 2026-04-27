"""Unit tests for the pure helpers in geometries/route.py.

The module has some big integrate-with-everything functions
(`get_multiple_ed`, `get_multi_drift_distance`) that exercise full
shapely + pyproj pipelines; these tests focus on the small helpers
(`get_best_utm`, `_get_ll_ur`, `cut`, `proj_point`, `get_angle`,
`create_line_grid`, `get_proj_transformer`).
"""
from __future__ import annotations

import sys
from math import isclose
from pathlib import Path

import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.route import (
    get_best_utm,
    _get_ll_ur,
    cut,
    proj_point,
    get_angle,
    create_line_grid,
    get_proj_transformer,
)


# ---------------------------------------------------------------------------
# _get_ll_ur
# ---------------------------------------------------------------------------

class TestGetLlUr:
    def test_line_string_bounds(self):
        line = LineString([(14.0, 55.0), (15.0, 56.0)])
        ll, ur = _get_ll_ur([line])
        assert ll == [14.0, 55.0]
        assert ur == [15.0, 56.0]

    def test_polygon_bounds(self):
        poly = box(10.0, 20.0, 30.0, 40.0)
        ll, ur = _get_ll_ur([poly])
        assert ll == [10.0, 20.0]
        assert ur == [30.0, 40.0]

    def test_mixed_objects_take_min_max(self):
        line = LineString([(10, 20), (30, 40)])
        poly = box(5, 15, 25, 35)
        ll, ur = _get_ll_ur([line, poly])
        assert ll == [5, 15] and ur == [30, 40]


# ---------------------------------------------------------------------------
# get_best_utm (pyproj lookup)
# ---------------------------------------------------------------------------

class TestGetBestUtm:
    def test_sweden_returns_utm_33n(self):
        line = LineString([(14.0, 55.0), (15.0, 55.0)])
        crs = get_best_utm([line])
        assert isinstance(crs, CRS)
        # UTM zone 33N for central Sweden.
        assert 'UTM' in crs.name.upper() or 'Zone 33' in crs.name

    def test_returns_crs_object(self):
        crs = get_best_utm([box(0.0, 0.0, 0.5, 0.5)])
        assert isinstance(crs, CRS)


# ---------------------------------------------------------------------------
# cut
# ---------------------------------------------------------------------------

class TestCut:
    def test_cut_beyond_length_returns_whole(self):
        line = LineString([(0, 0), (10, 0)])
        result = cut(line, 20)  # past the end
        assert len(result) == 1
        assert list(result[0].coords) == [(0, 0), (10, 0)]

    def test_cut_negative_returns_whole(self):
        line = LineString([(0, 0), (10, 0)])
        assert len(cut(line, -5)) == 1

    def test_cut_midpoint_splits_in_two(self):
        line = LineString([(0, 0), (10, 0)])
        parts = cut(line, 5)
        assert len(parts) == 2
        # First part ends at (5, 0); second starts at (5, 0).
        assert parts[0].coords[-1] == (5.0, 0.0)
        assert parts[1].coords[0] == (5.0, 0.0)

    def test_cut_exact_vertex_match(self):
        """Cutting at a coordinate that matches a vertex returns clean splits."""
        line = LineString([(0, 0), (5, 0), (10, 0)])
        parts = cut(line, 5)
        assert len(parts) == 2
        # Original first vertex list ends at (5, 0).
        assert parts[0].coords[-1] == (5.0, 0.0)


# ---------------------------------------------------------------------------
# proj_point
# ---------------------------------------------------------------------------

class TestProjPoint:
    def test_zero_bearing_moves_north(self):
        p = Point(0, 0)
        out = proj_point(p, distance=10.0, direction=0.0)
        assert isclose(out.x, 0.0, abs_tol=1e-9)
        assert isclose(out.y, 10.0, abs_tol=1e-9)

    def test_ninety_bearing_moves_east(self):
        p = Point(0, 0)
        out = proj_point(p, distance=10.0, direction=90.0)
        assert isclose(out.x, 10.0, abs_tol=1e-9)
        assert isclose(out.y, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# get_angle
# ---------------------------------------------------------------------------

class TestGetAngle:
    def test_east_angle_is_zero(self):
        """The function uses atan2(y_diff, x_diff) so a pure x-direction
        step (pt1=(0,0) -> pt2=(0,10), i.e. x_diff=10, y_diff=0) yields
        atan2(0, 10) = 0 degrees."""
        assert get_angle((0, 0), (0, 10)) == pytest.approx(0.0, abs=1e-6)

    def test_north_angle_is_ninety(self):
        # (y1, x1) -> (y2, x2): pt1=(0,0), pt2=(10,0) -> x_diff=0, y_diff=10
        # atan2(10, 0) = 90 deg
        assert get_angle((0, 0), (10, 0)) == pytest.approx(90.0, abs=1e-6)

    def test_roundtrip_pair(self):
        # Symmetric: reversing the pair flips the sign by 180.
        a = get_angle((0, 0), (10, 10))
        b = get_angle((10, 10), (0, 0))
        # Difference (mod 360) is 180.
        diff = (a - b) % 360
        assert diff == pytest.approx(180.0, abs=1e-6)


# ---------------------------------------------------------------------------
# create_line_grid
# ---------------------------------------------------------------------------

class TestCreateLineGrid:
    def test_returns_width_times_height_points(self):
        line = LineString([(0.0, 0.0), (1000.0, 0.0)])
        pts = create_line_grid(line, mu=0.0, std=50.0, width=10, height=10)
        # width * height points expected.
        assert len(pts) == 100
        for pt in pts:
            assert isinstance(pt, Point)


# ---------------------------------------------------------------------------
# get_proj_transformer
# ---------------------------------------------------------------------------

class TestGetProjTransformer:
    def test_returns_callable(self):
        line = LineString([(14.0, 55.0), (15.0, 55.0)])
        transformer = get_proj_transformer(line)
        assert callable(transformer)
        x, y = transformer(14.0, 55.0)
        # transformed to UTM meters, so coords are much larger than lon/lat.
        assert abs(x) > 100_000 and abs(y) > 1_000_000
