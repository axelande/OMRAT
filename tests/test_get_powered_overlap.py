"""Unit tests for the helpers in geometries/get_powered_overlap.py.

The ``PoweredOverlapVisualizer`` class is Qt + matplotlib-driven and
interactive; this file targets the standalone helpers used by
``compute/powered_model.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from shapely.geometry import (
    GeometryCollection, LineString, MultiPolygon, MultiPoint,
    MultiLineString, Point, Polygon, box,
)
from shapely.geometry.polygon import LinearRing

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.get_powered_overlap import (
    SimpleProjector, _parse_point, _project_wkt_geom,
    _weighted_avg_speed_knots, _leg_vectors, _make_band_polygon,
    _get_all_coords, _ray_hit_distance, _project_to_local,
    _plot_geom, _powered_na, _compute_cat2_with_shadows,
    _extract_edges_local,
    _build_legs_and_obstacles, _run_all_computations,
    find_closest_computation_index,
    N_RAYS, MAX_RANGE,
)


# ---------------------------------------------------------------------------
# SimpleProjector
# ---------------------------------------------------------------------------

class TestSimpleProjector:
    def test_origin_maps_to_zero(self):
        proj = SimpleProjector(lon_ref=14.0, lat_ref=55.0)
        x, y = proj.transform(14.0, 55.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_one_degree_east_in_meters(self):
        proj = SimpleProjector(lon_ref=14.0, lat_ref=55.0)
        x, _ = proj.transform(15.0, 55.0)
        # ~111_320 * cos(55°) ≈ 63_870 m
        assert 60_000 < x < 70_000


# ---------------------------------------------------------------------------
# _parse_point
# ---------------------------------------------------------------------------

class TestParsePoint:
    def test_two_tokens(self):
        assert _parse_point('14.0 55.0') == (14.0, 55.0)

    def test_extra_whitespace(self):
        assert _parse_point('  14.0   55.0  ') == (14.0, 55.0)


# ---------------------------------------------------------------------------
# _project_wkt_geom
# ---------------------------------------------------------------------------

class TestProjectWktGeom:
    def test_projects_polygon(self):
        proj = SimpleProjector(14.0, 55.0)
        poly = box(14.0, 55.0, 14.1, 55.1)
        out = _project_wkt_geom(poly, proj)
        # The lower-left corner should be at origin.
        minx, miny, maxx, maxy = out.bounds
        assert minx == pytest.approx(0.0, abs=1e-6)
        assert miny == pytest.approx(0.0, abs=1e-6)
        assert maxx > 1000  # ~6_300 m for 0.1° lon


# ---------------------------------------------------------------------------
# _weighted_avg_speed_knots
# ---------------------------------------------------------------------------

class TestWeightedAvgSpeedKnots:
    def test_no_freq_returns_zero(self):
        assert _weighted_avg_speed_knots({}) == 0.0

    def test_single_cell(self):
        d = {
            'Frequency (ships/year)': [[10.0]],
            'Speed (knots)': [[12.0]],
        }
        assert _weighted_avg_speed_knots(d) == 12.0

    def test_weighted_mean(self):
        d = {
            'Frequency (ships/year)': [[100.0, 50.0]],
            'Speed (knots)': [[10.0, 20.0]],
        }
        # (100*10 + 50*20) / (100+50) = 2000/150 = 13.33...
        assert _weighted_avg_speed_knots(d) == pytest.approx(13.333, abs=1e-2)

    def test_empty_string_skipped(self):
        d = {
            'Frequency (ships/year)': [['', 100.0]],
            'Speed (knots)': [['', 10.0]],
        }
        assert _weighted_avg_speed_knots(d) == 10.0

    def test_inf_values_skipped(self):
        d = {
            'Frequency (ships/year)': [[float('inf'), 100.0]],
            'Speed (knots)': [[10.0, 5.0]],
        }
        # The inf cell is rejected.
        assert _weighted_avg_speed_knots(d) == 5.0

    def test_garbage_string_skipped(self):
        d = {
            'Frequency (ships/year)': [['nope', 100.0]],
            'Speed (knots)': [[10.0, 5.0]],
        }
        assert _weighted_avg_speed_knots(d) == 5.0


# ---------------------------------------------------------------------------
# _leg_vectors
# ---------------------------------------------------------------------------

class TestLegVectors:
    def test_east_leg(self):
        u, n, L = _leg_vectors(np.array([0, 0]), np.array([10, 0]))
        assert u[0] == pytest.approx(1.0)
        assert n[1] == pytest.approx(1.0)
        assert L == 10.0

    def test_zero_length_returns_default(self):
        u, n, L = _leg_vectors(np.array([5, 5]), np.array([5, 5]))
        assert L == 0.0
        # u defaults to (1, 0).
        assert tuple(u) == (1.0, 0.0)


# ---------------------------------------------------------------------------
# _make_band_polygon
# ---------------------------------------------------------------------------

class TestMakeBandPolygon:
    def test_band_around_east_leg(self):
        poly = _make_band_polygon(
            start=np.array([0.0, 0.0]),
            end=np.array([100.0, 0.0]),
            mean_offset=0.0, sigma=10.0, n_sigma=3,
        )
        assert isinstance(poly, Polygon)
        # Half-width = 3*sigma = 30; total area = 100 * 60 = 6000.
        assert poly.area == pytest.approx(6000.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _get_all_coords
# ---------------------------------------------------------------------------

class TestGetAllCoords:
    def test_point(self):
        assert _get_all_coords(Point(1, 2)) == [(1.0, 2.0)]

    def test_linestring(self):
        coords = _get_all_coords(LineString([(0, 0), (1, 1)]))
        assert coords == [(0.0, 0.0), (1.0, 1.0)]

    def test_polygon_returns_exterior(self):
        coords = _get_all_coords(box(0, 0, 1, 1))
        # 5 points (closed ring).
        assert len(coords) == 5

    def test_multipoint(self):
        coords = _get_all_coords(MultiPoint([(0, 0), (1, 1)]))
        assert (0.0, 0.0) in coords and (1.0, 1.0) in coords

    def test_multilinestring(self):
        mls = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
        coords = _get_all_coords(mls)
        assert len(coords) == 4

    def test_multipolygon(self):
        mp = MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)])
        coords = _get_all_coords(mp)
        assert len(coords) == 10

    def test_geometry_collection(self):
        gc = GeometryCollection([Point(0, 0), LineString([(1, 1), (2, 2)])])
        coords = _get_all_coords(gc)
        assert (0.0, 0.0) in coords
        assert (1.0, 1.0) in coords

    def test_linear_ring(self):
        lr = LinearRing([(0, 0), (1, 0), (1, 1), (0, 1)])
        coords = _get_all_coords(lr)
        assert len(coords) >= 4

    def test_empty_returns_empty(self):
        assert _get_all_coords(Polygon()) == []


# ---------------------------------------------------------------------------
# _ray_hit_distance
# ---------------------------------------------------------------------------

class TestRayHitDistance:
    def test_hit_distance(self):
        # Ray going east from origin hits a box at x=10.
        obs = box(10, -1, 12, 1)
        d = _ray_hit_distance(
            origin=np.array([0.0, 0.0]),
            direction=np.array([1.0, 0.0]),
            max_range=100.0,
            obstacle_geom=obs,
        )
        assert d == pytest.approx(10.0, abs=1e-6)

    def test_no_intersection_returns_none(self):
        # Box far away from ray.
        obs = box(100, 100, 110, 110)
        d = _ray_hit_distance(
            origin=np.array([0.0, 0.0]),
            direction=np.array([1.0, 0.0]),
            max_range=50.0,
            obstacle_geom=obs,
        )
        assert d is None

    def test_obstacle_behind_origin_returns_none(self):
        # Obstacle behind the ray origin: ray going east, obstacle at x=-10.
        # The ray.intersects(obstacle) is False so returns None.
        obs = box(-12, -1, -10, 1)
        d = _ray_hit_distance(
            origin=np.array([0.0, 0.0]),
            direction=np.array([1.0, 0.0]),
            max_range=100.0,
            obstacle_geom=obs,
        )
        assert d is None


# ---------------------------------------------------------------------------
# _project_to_local
# ---------------------------------------------------------------------------

class TestProjectToLocal:
    def test_axis_aligned(self):
        # Origin at (0,0), along=east, perp=north.
        out = _project_to_local(
            geom=Point(10, 5),
            origin=np.array([0.0, 0.0]),
            along_dir=np.array([1.0, 0.0]),
            perp_dir=np.array([0.0, 1.0]),
        )
        assert out.x == pytest.approx(10.0)
        assert out.y == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _plot_geom
# ---------------------------------------------------------------------------

class TestPlotGeom:
    def test_polygon_plotted(self, qgis_iface):
        from matplotlib.figure import Figure
        ax = Figure().subplots()
        _plot_geom(ax, box(0, 0, 1, 1), color='red', label='x')
        # ax.patches grew (fill draws a Polygon patch).
        assert len(ax.patches) == 1

    def test_multipolygon_plotted(self, qgis_iface):
        from matplotlib.figure import Figure
        ax = Figure().subplots()
        mp = MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)])
        _plot_geom(ax, mp, color='blue', label='m')
        assert len(ax.patches) == 2


# ---------------------------------------------------------------------------
# _powered_na
# ---------------------------------------------------------------------------

class TestPoweredNa:
    def test_zero_recovery_returns_zero(self):
        assert _powered_na(distance=100.0, mean_time=0.0, ship_speed=10.0) == 0.0
        assert _powered_na(distance=100.0, mean_time=180.0, ship_speed=0.0) == 0.0

    def test_zero_distance_returns_one(self):
        assert _powered_na(distance=0.0, mean_time=180.0, ship_speed=5.0) == 1.0

    def test_finite_decay(self):
        # exp(-100 / (180 * 5)) = exp(-0.111) ~ 0.895
        v = _powered_na(distance=100.0, mean_time=180.0, ship_speed=5.0)
        assert 0.8 < v < 0.95


# ---------------------------------------------------------------------------
# _compute_cat2_with_shadows
# ---------------------------------------------------------------------------

class TestComputeCat2WithShadows:
    def test_no_obstacles_returns_empty(self):
        summaries, ray_data, offsets, pdf_vals = _compute_cat2_with_shadows(
            turn_pt=np.array([0.0, 0.0]),
            ext_dir=np.array([1.0, 0.0]),
            perp=np.array([0.0, 1.0]),
            mean_offset=0.0, sigma=10.0,
            ai=180.0, speed_ms=5.0,
            obstacles=[],
        )
        assert summaries == {}
        # ray_data has N_RAYS entries (no obstacle hit).
        assert len(ray_data) == len(offsets)

    def test_single_obstacle_hit(self):
        # Box directly ahead of the turn point.
        obs_geom = box(50, -50, 100, 50)
        summaries, ray_data, offsets, pdf_vals = _compute_cat2_with_shadows(
            turn_pt=np.array([0.0, 0.0]),
            ext_dir=np.array([1.0, 0.0]),
            perp=np.array([0.0, 1.0]),
            mean_offset=0.0, sigma=10.0,
            ai=180.0, speed_ms=5.0,
            obstacles=[({'id': 'obs1', 'geom': obs_geom}, 'object')],
        )
        # At least one ray hit the obstacle.
        assert summaries
        key = ('object', 'obs1')
        assert key in summaries
        assert summaries[key]['mass'] > 0


# ---------------------------------------------------------------------------
# _build_legs_and_obstacles
# ---------------------------------------------------------------------------

class TestBuildLegsAndObstacles:
    def _proj(self):
        return SimpleProjector(14.0, 55.0)

    def test_grounding_mode_only_depths(self):
        data = {
            'segment_data': {
                '1': {'Start_Point': '14.0 55.0', 'End_Point': '14.1 55.0',
                      'Dirs': ['E', 'W'], 'ai1': 180, 'ai2': 180,
                      'mean1_1': 0, 'std1_1': 100,
                      'mean2_1': 0, 'std2_1': 100, 'Leg_name': 'L1'},
            },
            'depths': [['d1', '5', 'POLYGON((14.05 55.0, 14.06 55.0, 14.06 55.01, 14.05 55.01, 14.05 55.0))']],
            'objects': [['s1', '20', 'POLYGON((14.07 55.0, 14.08 55.0, 14.08 55.01, 14.07 55.01, 14.07 55.0))']],
            'traffic_data': {},
        }
        legs, obs, dg, dgd, og = _build_legs_and_obstacles(
            data, self._proj(), mode='grounding', max_draft=15.0)
        assert len(legs) == 1
        # Only depth obstacles in grounding mode.
        assert all(kind == 'depth' for _, kind in obs)

    def test_allision_mode_only_objects(self):
        data = {
            'segment_data': {
                '1': {'Start_Point': '14.0 55.0', 'End_Point': '14.1 55.0',
                      'Dirs': ['E', 'W'], 'ai1': 180, 'ai2': 180,
                      'mean1_1': 0, 'std1_1': 100,
                      'mean2_1': 0, 'std2_1': 100, 'Leg_name': 'L1'},
            },
            'depths': [['d1', '5', 'POLYGON((14.05 55.0, 14.06 55.0, 14.06 55.01, 14.05 55.01, 14.05 55.0))']],
            'objects': [['s1', '20', 'POLYGON((14.07 55.0, 14.08 55.0, 14.08 55.01, 14.07 55.01, 14.07 55.0))']],
            'traffic_data': {},
        }
        legs, obs, dg, dgd, og = _build_legs_and_obstacles(
            data, self._proj(), mode='allision', max_draft=0)
        assert all(kind == 'object' for _, kind in obs)

    def test_both_mode_has_both(self):
        data = {
            'segment_data': {
                '1': {'Start_Point': '14.0 55.0', 'End_Point': '14.1 55.0',
                      'Dirs': ['E', 'W'], 'ai1': 180, 'ai2': 180,
                      'mean1_1': 0, 'std1_1': 100,
                      'mean2_1': 0, 'std2_1': 100, 'Leg_name': 'L1'},
            },
            'depths': [['d1', '5', 'POLYGON((14.05 55.0, 14.06 55.0, 14.06 55.01, 14.05 55.01, 14.05 55.0))']],
            'objects': [['s1', '20', 'POLYGON((14.07 55.0, 14.08 55.0, 14.08 55.01, 14.07 55.01, 14.07 55.0))']],
            'traffic_data': {},
        }
        legs, obs, dg, dgd, og = _build_legs_and_obstacles(
            data, self._proj(), mode='both', max_draft=15.0)
        kinds = {k for _, k in obs}
        assert kinds == {'depth', 'object'}

    def test_malformed_segment_skipped(self):
        data = {
            'segment_data': {
                'bad': {'Start_Point': 'nonsense', 'End_Point': '14.1 55.0'},
            },
            'depths': [], 'objects': [], 'traffic_data': {},
        }
        legs, obs, dg, dgd, og = _build_legs_and_obstacles(
            data, self._proj(), mode='grounding', max_draft=10.0)
        assert legs == {}

    def test_malformed_depth_swallowed(self):
        data = {
            'segment_data': {},
            'depths': [['d1', 'not-a-number', 'NOT WKT']],
            'objects': [], 'traffic_data': {},
        }
        legs, obs, dg, dgd, og = _build_legs_and_obstacles(
            data, self._proj(), mode='grounding', max_draft=10.0)
        assert dg == []


# ---------------------------------------------------------------------------
# _extract_edges_local -- helper for the vectorised Cat II ray-cast
# ---------------------------------------------------------------------------

class TestExtractEdgesLocal:
    def test_box_in_aligned_frame(self):
        from shapely.geometry import box as shp_box
        # Ray leg: origin (0,0), along = +x, perp = +y.
        # A box at x=500..600, y=-50..50 should yield 4 closing edges whose
        # lateral range is [-50, 50] and along range is [500, 600].
        edges = _extract_edges_local(
            shp_box(500, -50, 600, 50),
            turn_pt=np.array([0.0, 0.0]),
            along_dir=np.array([1.0, 0.0]),
            perp_dir=np.array([0.0, 1.0]),
        )
        assert edges.shape == (4, 2, 2)
        # Lateral coords stay in [-50, 50]; along coords in [500, 600].
        lateral_all = edges[..., 1]
        along_all = edges[..., 0]
        assert lateral_all.min() == pytest.approx(-50.0)
        assert lateral_all.max() == pytest.approx(50.0)
        assert along_all.min() == pytest.approx(500.0)
        assert along_all.max() == pytest.approx(600.0)

    def test_empty_geometry_returns_none(self):
        from shapely.geometry import Polygon as ShpPolygon
        assert _extract_edges_local(
            ShpPolygon(),
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]),
        ) is None

    def test_point_has_no_edges(self):
        """A Point obstacle is measure-zero; skip it."""
        from shapely.geometry import Point
        assert _extract_edges_local(
            Point(100, 0),
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]),
        ) is None

    def test_multipolygon_concatenates(self):
        from shapely.geometry import box as shp_box, MultiPolygon
        mp = MultiPolygon([shp_box(100, -10, 200, 10), shp_box(300, -10, 400, 10)])
        edges = _extract_edges_local(
            mp,
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]),
        )
        # Two boxes -> 4 + 4 = 8 edges.
        assert edges.shape == (8, 2, 2)


# ---------------------------------------------------------------------------
# _compute_cat2_with_shadows -- vectorised vs reference implementation
# ---------------------------------------------------------------------------

class TestComputeCat2Vectorised:
    """Lock in the vectorised ray-cast against a plain ray-by-ray reference.

    The vectorised implementation must produce identical hit counts, masses,
    and mean distances to a straightforward scalar loop using shapely
    ray-polygon intersections.  This guards against future refactors
    silently changing the shadow cascade.
    """

    def _reference_scalar(self, turn_pt, ext_dir, perp, mean_offset, sigma,
                          ai, speed_ms, obstacles):
        """Ray-by-ray, shapely-backed reference (old implementation)."""
        from collections import defaultdict
        from math import exp
        from scipy.stats import norm

        offsets = np.linspace(mean_offset - 4 * sigma, mean_offset + 4 * sigma, N_RAYS)
        dx = offsets[1] - offsets[0]
        masses = norm.pdf(offsets, mean_offset, sigma) * dx
        recovery = ai * speed_ms

        accum: dict = defaultdict(lambda: {
            'mass': 0.0, 'weighted_dist': 0.0, 'p_integral': 0.0, 'n_rays': 0,
        })
        for off, m_i in zip(offsets, masses):
            ray_origin = turn_pt + off * perp
            best_d = float('inf')
            best_key = None
            for obs, kind in obstacles:
                d = _ray_hit_distance(ray_origin, ext_dir, MAX_RANGE, obs['geom'])
                if d is not None and 0 < d < best_d:
                    best_d, best_key = d, (kind, obs['id'])
            if best_key is not None:
                a = accum[best_key]
                a['mass'] += m_i
                a['weighted_dist'] += m_i * best_d
                if recovery > 0:
                    a['p_integral'] += m_i * exp(-best_d / recovery)
                a['n_rays'] += 1
        return accum

    def test_matches_reference_on_shadowed_scene(self):
        from shapely.geometry import box as shp_box, Polygon as ShpPolygon
        turn_pt = np.array([0.0, 0.0])
        ext_dir = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])
        # Mix of boxes (casts shadow) and a triangle (non-rectangular).
        obstacles = [
            ({'id': 'near', 'geom': shp_box(500, -50, 700, 50)}, 'object'),
            ({'id': 'far', 'geom': shp_box(2000, -200, 2500, 200)}, 'depth'),
            ({'id': 'side', 'geom': shp_box(800, 100, 1000, 300)}, 'object'),
            ({'id': 'shadow', 'geom': shp_box(1200, -100, 1400, 100)}, 'depth'),
            ({'id': 'tri', 'geom': ShpPolygon([(3000, -100), (3500, 0), (3000, 100)])}, 'object'),
        ]

        summaries, ray_data, _, _ = _compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, 0.0, 100.0, 180, 5.0, obstacles,
        )
        ref = self._reference_scalar(
            turn_pt, ext_dir, perp, 0.0, 100.0, 180, 5.0, obstacles,
        )

        # Same set of hit obstacles and same n_rays/mass/mean per obstacle.
        assert set(summaries.keys()) == set(ref.keys())
        for key, s in summaries.items():
            r = ref[key]
            assert s['n_rays'] == r['n_rays']
            assert s['mass'] == pytest.approx(r['mass'])
            ref_mean = r['weighted_dist'] / r['mass'] if r['mass'] > 0 else 0
            assert s['mean_dist'] == pytest.approx(ref_mean)

    def test_no_obstacles_returns_empty(self):
        summaries, ray_data, offsets, pdf = _compute_cat2_with_shadows(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]),
            0.0, 50.0, 180, 5.0, [],
        )
        assert summaries == {}
        assert len(ray_data) == N_RAYS
        # Every ray recorded as a miss.
        assert all(r[2] is None and r[3] is None for r in ray_data)


# ---------------------------------------------------------------------------
# _run_all_computations
# ---------------------------------------------------------------------------

class TestRunAllComputations:
    def test_zero_speed_dir_skipped(self):
        legs = {
            '1': {
                'start': np.array([0.0, 0.0]),
                'end': np.array([1000.0, 0.0]),
                'name': 'L1',
                'start_wkt': '0 0',
                'end_wkt': '1000 0',
                'dirs': [
                    {'name': 'E', 'speed_ms': 0.0, 'ai': 180,
                     'mean': 0, 'std': 100, 'speed_kn': 0},
                    {'name': 'W', 'speed_ms': 0.0, 'ai': 180,
                     'mean': 0, 'std': 100, 'speed_kn': 0},
                ],
            },
        }
        comps = _run_all_computations(legs, [])
        assert comps == []

    def test_with_obstacle_produces_computation(self):
        legs = {
            '1': {
                'start': np.array([0.0, 0.0]),
                'end': np.array([100.0, 0.0]),
                'name': 'L1',
                'start_wkt': '0 0',
                'end_wkt': '100 0',
                'dirs': [
                    {'name': 'E', 'speed_ms': 5.0, 'ai': 180,
                     'mean': 0, 'std': 10, 'speed_kn': 10},
                ],
            },
        }
        # Obstacle ahead (east) of the turn point at end of leg.
        obs = {'id': 'obs', 'geom': box(150, -50, 250, 50)}
        comps = _run_all_computations(legs, [(obs, 'object')])
        assert len(comps) == 1
        assert comps[0]['seg_id'] == '1'
        assert comps[0]['dir_idx'] == 0


# ---------------------------------------------------------------------------
# find_closest_computation_index  (refactored helper)
# ---------------------------------------------------------------------------

class TestFindClosestComputationIndex:
    def _comps(self):
        return [
            {'turn_pt': np.array([0.0, 0.0]), 'name': 'A'},
            {'turn_pt': np.array([100.0, 0.0]), 'name': 'B'},
            {'turn_pt': np.array([0.0, 100.0]), 'name': 'C'},
        ]

    def test_within_threshold_returns_closest(self):
        idx = find_closest_computation_index(
            click_xy=(105.0, 5.0),  # closest to B
            computations=self._comps(),
            threshold=20.0,
        )
        assert idx == 1

    def test_outside_threshold_returns_none(self):
        idx = find_closest_computation_index(
            click_xy=(500.0, 500.0),
            computations=self._comps(),
            threshold=20.0,
        )
        assert idx is None

    def test_empty_comps_returns_none(self):
        assert find_closest_computation_index(
            click_xy=(0.0, 0.0), computations=[], threshold=100.0,
        ) is None


# ---------------------------------------------------------------------------
# PoweredOverlapVisualizer
# ---------------------------------------------------------------------------

class TestPoweredOverlapVisualizer:
    def _make_visualizer(self, qgis_iface):
        """Build a visualizer with a single leg + one obstacle hit."""
        from geometries.get_powered_overlap import PoweredOverlapVisualizer
        from matplotlib.figure import Figure

        # Build leg + obstacle.
        legs = {
            '1': {
                'start': np.array([0.0, 0.0]),
                'end': np.array([1_000.0, 0.0]),
                'name': 'L1',
                'start_wkt': '0 0',
                'end_wkt': '1000 0',
                'dirs': [
                    {'name': 'East going', 'speed_ms': 5.0, 'ai': 180,
                     'mean': 0, 'std': 50, 'speed_kn': 10},
                    {'name': 'West going', 'speed_ms': 4.0, 'ai': 180,
                     'mean': 0, 'std': 50, 'speed_kn': 8},
                ],
            },
        }
        obs_dict = {'id': 'obs1', 'geom': box(1_500, -100, 2_000, 100)}
        depth_dict = {'id': 'd1', 'depth': 5.0,
                      'geom': box(-1_500, -100, -1_000, 100)}
        all_obs = [(obs_dict, 'object'), (depth_dict, 'depth')]
        depth_geoms = [depth_dict]
        depth_geoms_deep = [{'id': 'deep', 'depth': 100.0, 'geom': box(0, 500, 1000, 600)}]
        object_geoms = [obs_dict]
        comps = _run_all_computations(legs, all_obs)

        fig = Figure()
        # gridspec needs at least one axes to attach.
        ax_overview = fig.add_subplot(2, 2, (1, 2))
        ax_detail = fig.add_subplot(2, 2, 3)
        ax_waterfall = fig.add_subplot(2, 2, 4)
        axes = {
            'overview': ax_overview, 'detail': ax_detail,
            'waterfall': ax_waterfall,
        }
        return PoweredOverlapVisualizer(
            fig=fig, axes=axes, legs=legs, all_obstacles=all_obs,
            depth_geoms=depth_geoms, depth_geoms_deep=depth_geoms_deep,
            object_geoms=object_geoms, computations=comps, mode='allision',
        )

    def test_init_stores_inputs(self, qgis_iface):
        v = self._make_visualizer(qgis_iface)
        assert v.mode == 'allision'
        assert v._selected_comp_idx is None
        # Color maps populated.
        assert len(v.dir_colors) == 2
        assert len(v.leg_colors) == 1

    def test_run_visualization_draws_overview_and_selects(self, qgis_iface):
        v = self._make_visualizer(qgis_iface)
        v.run_visualization()
        # Overview drew at least one line (leg).
        assert len(v.axes['overview'].get_lines()) >= 1
        # Default computation selected if computations exist.
        if v.computations:
            assert v._selected_comp_idx == 0

    def test_select_computation_out_of_range_noop(self, qgis_iface):
        v = self._make_visualizer(qgis_iface)
        v._select_computation(99)
        assert v._selected_comp_idx is None
        v._select_computation(-1)
        assert v._selected_comp_idx is None

    def test_select_computation_redraws_detail(self, qgis_iface):
        v = self._make_visualizer(qgis_iface)
        v.run_visualization()
        # Force a redraw on a (possibly different) computation.
        if len(v.computations) > 0:
            v._select_computation(0)
            # Detail axis has at least the centerline / annotations.
            assert v._selected_comp_idx == 0

    def test_on_overview_click_wrong_axes(self, qgis_iface):
        from types import SimpleNamespace
        v = self._make_visualizer(qgis_iface)
        v.run_visualization()
        original_idx = v._selected_comp_idx
        evt = SimpleNamespace(inaxes=v.axes['detail'], xdata=10, ydata=10)
        v._on_overview_click(evt)
        assert v._selected_comp_idx == original_idx

    def test_on_overview_click_no_xy(self, qgis_iface):
        from types import SimpleNamespace
        v = self._make_visualizer(qgis_iface)
        v.run_visualization()
        evt = SimpleNamespace(
            inaxes=v.axes['overview'], xdata=None, ydata=None,
        )
        # Should not raise.
        v._on_overview_click(evt)

    def test_on_overview_click_near_turn_point_selects(self, qgis_iface):
        from types import SimpleNamespace
        v = self._make_visualizer(qgis_iface)
        v.run_visualization()
        if not v.computations:
            return  # nothing to click
        tp = v.computations[0]['turn_pt']
        evt = SimpleNamespace(
            inaxes=v.axes['overview'],
            xdata=float(tp[0]) + 1.0, ydata=float(tp[1]) + 1.0,
        )
        v._on_overview_click(evt)
        assert v._selected_comp_idx == 0

    def test_no_computations_run_visualization_safe(self, qgis_iface):
        """An empty computations list should still allow run_visualization."""
        from geometries.get_powered_overlap import PoweredOverlapVisualizer
        from matplotlib.figure import Figure
        fig = Figure()
        ax_overview = fig.add_subplot(2, 2, (1, 2))
        ax_detail = fig.add_subplot(2, 2, 3)
        ax_waterfall = fig.add_subplot(2, 2, 4)
        axes = {'overview': ax_overview, 'detail': ax_detail, 'waterfall': ax_waterfall}
        v = PoweredOverlapVisualizer(
            fig=fig, axes=axes,
            legs={}, all_obstacles=[],
            depth_geoms=[], depth_geoms_deep=[], object_geoms=[],
            computations=[], mode='grounding',
        )
        v.run_visualization()  # no raise
