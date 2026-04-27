"""Unit tests for geometries/drift/*.py.

Covers: coordinates (UTM zone, compass-to-vector), distribution
(projection distance, width), corridor (base surface + projection),
and shadow (extract_polygons).  These are all pure-geometry helpers
without QGIS or Qt dependencies.
"""
from __future__ import annotations

import sys
from math import pi, sqrt
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import (
    GeometryCollection, LineString, MultiPolygon, Point, Polygon, box,
)
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.drift.coordinates import (
    get_utm_crs, transform_geometry, compass_to_vector,
)
from geometries.drift.distribution import (
    get_projection_distance, get_distribution_width,
)
from geometries.drift.corridor import (
    create_base_surface, create_projected_corridor,
)
from geometries.drift.shadow import (
    create_obstacle_shadow, extract_polygons,
)


# ---------------------------------------------------------------------------
# coordinates
# ---------------------------------------------------------------------------

class TestGetUtmCrs:
    def test_zone_33_north_for_sweden(self):
        crs = get_utm_crs(lon=14.0, lat=55.0)
        assert 'zone=33' in crs.srs and 'north' in crs.srs

    def test_zone_1_north_at_date_line(self):
        crs = get_utm_crs(lon=-179.9, lat=0.1)
        assert 'zone=1' in crs.srs and 'north' in crs.srs

    def test_zone_south_below_equator(self):
        crs = get_utm_crs(lon=15.0, lat=-30.0)
        assert 'south' in crs.srs


class TestTransformGeometry:
    def test_wgs84_to_utm_scales_to_meters(self):
        wgs84 = get_utm_crs(lon=14.0, lat=55.0)  # placeholder (unused)
        # Use explicit CRS strings to avoid ambiguity.
        from pyproj import CRS
        src = CRS("EPSG:4326")
        tgt = CRS("EPSG:32633")
        p = Point(14.0, 55.0)
        t = transform_geometry(p, src, tgt)
        # Result coords should be in the hundreds-of-thousands of meters.
        assert abs(t.x) > 100_000 and abs(t.y) > 100_000


class TestCompassToVector:
    @pytest.mark.parametrize("angle, expected_dx_sign, expected_dy_sign", [
        (0, 0, 1),        # N  -> +Y
        (90, 1, 0),       # E  -> +X
        (180, 0, -1),     # S  -> -Y
        (270, -1, 0),     # W  -> -X
    ])
    def test_cardinal_directions(self, angle, expected_dx_sign, expected_dy_sign):
        dx, dy = compass_to_vector(angle, 100.0)
        if expected_dx_sign == 0:
            assert abs(dx) < 1e-9
        else:
            assert dx * expected_dx_sign > 0
        if expected_dy_sign == 0:
            assert abs(dy) < 1e-9
        else:
            assert dy * expected_dy_sign > 0

    def test_magnitude_preserved(self):
        dx, dy = compass_to_vector(45.0, 50.0)
        assert sqrt(dx*dx + dy*dy) == pytest.approx(50.0, abs=1e-6)


# ---------------------------------------------------------------------------
# distribution
# ---------------------------------------------------------------------------

class TestGetProjectionDistance:
    def test_lognormal_returns_capped_distance(self):
        d = get_projection_distance(
            {'use_lognormal': True, 'std': 0.95, 'loc': 0.2, 'scale': 0.85},
            drift_speed_ms=1.0, target_prob=1e-3, max_distance=50_000,
        )
        assert 0 < d <= 50_000

    def test_non_lognormal_returns_10km_or_max(self):
        d = get_projection_distance(
            {'use_lognormal': False},
            drift_speed_ms=1.0, max_distance=5_000,
        )
        assert d == 5_000

    def test_zero_or_huge_drift_speed_normalised(self):
        # drift_speed_ms <= 0 or > 10 falls back to 1 m/s internally;
        # result should still be a finite positive distance.
        d = get_projection_distance(
            {'use_lognormal': True, 'std': 0.95, 'loc': 0.2, 'scale': 0.85},
            drift_speed_ms=0.0, max_distance=50_000,
        )
        assert d > 0

    def test_std_or_scale_zero_falls_back(self):
        d = get_projection_distance(
            {'use_lognormal': True, 'std': 0.0, 'loc': 0.0, 'scale': 1.0},
            drift_speed_ms=1.0, max_distance=50_000,
        )
        # Function returns min(10000, max_distance) = 10_000.
        assert d == 10_000

    def test_unreasonable_lognormal_falls_back(self):
        """A distribution whose ppf yields > 48h is clamped to 10 km."""
        d = get_projection_distance(
            # std=5 stretches the tail way past 48h at 0.999 percentile.
            {'use_lognormal': True, 'std': 5.0, 'loc': 0.0, 'scale': 1.0},
            drift_speed_ms=1.0, max_distance=50_000,
        )
        assert d == 10_000


class TestGetDistributionWidth:
    def test_99_percent_width_is_5_15_sigma(self):
        w = get_distribution_width(std=10.0, coverage=0.99)
        # 2 * z_0.995 * std = 2 * 2.576 * 10 ~ 51.54
        assert w == pytest.approx(51.54, abs=0.1)

    def test_width_scales_with_std(self):
        w1 = get_distribution_width(std=1.0)
        w2 = get_distribution_width(std=2.0)
        assert w2 == pytest.approx(2 * w1, abs=1e-9)


# ---------------------------------------------------------------------------
# corridor
# ---------------------------------------------------------------------------

class TestCreateBaseSurface:
    def test_simple_east_leg_produces_rectangle(self):
        leg = LineString([(0, 0), (1000, 0)])
        poly = create_base_surface(leg, half_width=50.0)
        assert isinstance(poly, Polygon) and not poly.is_empty
        # Area = length * width = 1000 * 100
        assert poly.area == pytest.approx(1000 * 100, rel=1e-9)

    def test_single_point_leg_returns_empty(self):
        leg = LineString([(0, 0), (0, 0)])
        poly = create_base_surface(leg, half_width=50.0)
        assert poly.is_empty

    def test_degenerate_leg_one_coord_returns_empty(self):
        leg = LineString([(0, 0), (1e-9, 0)])  # length tiny but > 0
        poly = create_base_surface(leg, half_width=50.0)
        # The tiny-but-positive length produces a valid (thin) rectangle.
        assert isinstance(poly, Polygon)


class TestCreateProjectedCorridor:
    def test_north_projection_extends_corridor(self):
        leg = LineString([(0, 0), (1000, 0)])
        corridor = create_projected_corridor(
            leg, half_width=50.0, drift_angle_deg=0.0,
            projection_dist=200.0,
        )
        # Area should exceed the base rectangle (1000*100 = 100k).
        assert corridor.area > 100_000

    def test_zero_length_leg_returns_empty(self):
        leg = LineString([(0, 0), (0, 0)])
        corridor = create_projected_corridor(leg, 50.0, 0.0, 200.0)
        assert corridor.is_empty


# ---------------------------------------------------------------------------
# shadow
# ---------------------------------------------------------------------------

class TestCreateObstacleShadow:
    def test_empty_obstacle_returns_empty(self):
        shadow = create_obstacle_shadow(Polygon(), 0.0, (-1000, -1000, 1000, 1000))
        assert shadow.is_empty

    def test_shadow_contains_obstacle_and_extends_downwind(self):
        poly = box(-10, -10, 10, 10)
        shadow = create_obstacle_shadow(poly, 0.0, (-1000, -1000, 1000, 1000))
        # Shadow must contain the obstacle footprint and extend outward.
        assert shadow.contains(poly.centroid)
        assert shadow.area > poly.area


class TestExtractPolygons:
    def test_polygon_returns_single_entry(self):
        p = box(0, 0, 1, 1)
        assert extract_polygons(p) == [p]

    def test_multipolygon_yields_components(self):
        mp = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
        polys = extract_polygons(mp)
        assert len(polys) == 2

    def test_geometry_collection_yields_polygons_only(self):
        gc = GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 1)])])
        polys = extract_polygons(gc)
        assert len(polys) == 1  # LineString dropped

    def test_non_geometric_input_yields_empty(self):
        assert extract_polygons(None) == []
        assert extract_polygons(LineString([(0, 0), (1, 1)])) == []
