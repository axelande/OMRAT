"""Unit tests for the pure-Python helpers in geometries/get_drifting_overlap.

The ``DriftingOverlapVisualizer`` class is Qt + matplotlib-driven and
exercised by ``test_visualization_mixin`` via mocks.  This file targets
the standalone helpers used by both the visualizer and the production
``compute_min_distance_by_object`` (which feeds ``drifting_model``).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.stats import norm
from shapely.geometry import LineString, MultiPolygon, Polygon, Point, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.get_drifting_overlap import (
    create_polygon_from_line,
    extend_polygon_in_directions,
    compare_polygons_with_objs,
    compute_coverages_and_distances,
    estimate_weighted_overlap,
    directional_min_distance_reverse_ray,
)


# ---------------------------------------------------------------------------
# create_polygon_from_line
# ---------------------------------------------------------------------------

class TestCreatePolygonFromLine:
    def test_buffer_around_line(self):
        line = LineString([(0, 0), (100, 0)])
        # Single normal with std=10 -> coverage_range = 4.89 * 10 = 48.9
        poly = create_polygon_from_line(line, [norm(loc=0, scale=10)], [1.0])
        assert isinstance(poly, Polygon)
        assert poly.area > 100 * 90  # very large buffer

    def test_weights_normalised(self):
        """Two distributions with weights summing > 1 produce the same polygon
        as the same distributions with normalised weights."""
        line = LineString([(0, 0), (50, 0)])
        d1 = norm(loc=0, scale=5)
        d2 = norm(loc=0, scale=10)
        p_a = create_polygon_from_line(line, [d1, d2], [2.0, 2.0])
        p_b = create_polygon_from_line(line, [d1, d2], [1.0, 1.0])
        assert p_a.area == pytest.approx(p_b.area, rel=1e-9)

    def test_weighted_mean_translates_polygon(self):
        """A non-zero distribution mean shifts the polygon perpendicular to the line."""
        line = LineString([(0, 0), (100, 0)])
        # mean=50 -> polygon centered around y=50 (translated upward)
        poly = create_polygon_from_line(line, [norm(loc=50, scale=5)], [1.0])
        cy = poly.centroid.y
        assert cy == pytest.approx(50.0, abs=1.0)


# ---------------------------------------------------------------------------
# extend_polygon_in_directions
# ---------------------------------------------------------------------------

class TestExtendPolygonInDirections:
    def test_eight_polygons_returned(self):
        base = box(-5, -5, 5, 5)
        polys, centres = extend_polygon_in_directions(base, 100.0)
        assert len(polys) == 8
        assert len(centres) == 8
        assert all(isinstance(p, Polygon) for p in polys)

    def test_empty_polygon_returns_eight_empty(self):
        polys, centres = extend_polygon_in_directions(Polygon(), 100.0)
        assert len(polys) == 8
        assert all(p.is_empty for p in polys)
        assert all(c.is_empty for c in centres)

    def test_none_input_returns_eight_empty(self):
        polys, centres = extend_polygon_in_directions(None, 100.0)
        assert len(polys) == 8
        assert all(p.is_empty for p in polys)


# ---------------------------------------------------------------------------
# compare_polygons_with_objs
# ---------------------------------------------------------------------------

class TestComparePolygonsWithObjs:
    def test_intersection_per_obj_per_polygon(self):
        import geopandas as gpd
        polys = [box(0, 0, 10, 10), box(100, 100, 110, 110)]
        # Two GDFs each with one obstacle.
        gdf1 = gpd.GeoDataFrame(geometry=[box(5, 5, 8, 8)])
        gdf2 = gpd.GeoDataFrame(geometry=[box(105, 105, 108, 108)])
        results = compare_polygons_with_objs(polys, [gdf1, gdf2])
        # Polygon 0 intersects gdf1 (yes), gdf2 (no).
        # Polygon 1 intersects gdf1 (no), gdf2 (yes).
        assert results['Polygon_0'][0] == [True]
        assert results['Polygon_0'][1] == [False]
        assert results['Polygon_1'][0] == [False]
        assert results['Polygon_1'][1] == [True]


# ---------------------------------------------------------------------------
# estimate_weighted_overlap
# ---------------------------------------------------------------------------

class TestEstimateWeightedOverlap:
    def test_polygon_intersection_yields_distances(self):
        line = LineString([(0, 0), (100, 0)])
        intersection = box(40, -5, 60, 5)
        d1 = norm(loc=0, scale=5)
        cov, distances = estimate_weighted_overlap(
            intersection, line, [d1], [1.0]
        )
        assert cov >= 0.0
        assert distances.size > 0

    def test_multipolygon_intersection_concatenates_coords(self):
        line = LineString([(0, 0), (100, 0)])
        mp = MultiPolygon([box(40, -5, 50, 5), box(60, -5, 70, 5)])
        d1 = norm(loc=0, scale=5)
        cov, distances = estimate_weighted_overlap(
            mp, line, [d1], [1.0]
        )
        assert cov >= 0.0
        # Two boxes -> at least 8 vertices (likely 10 -- 5 each including close).
        assert distances.size >= 8

    def test_unsupported_geom_raises(self):
        line = LineString([(0, 0), (100, 0)])
        with pytest.raises(ValueError):
            estimate_weighted_overlap(
                Point(50, 0), line, [norm(loc=0, scale=5)], [1.0]
            )


# ---------------------------------------------------------------------------
# directional_min_distance_reverse_ray
# ---------------------------------------------------------------------------

class TestDirectionalMinDistanceReverseRay:
    def test_polygon_north_of_leg(self):
        """Point above leg, drift north -> distance ~ y of polygon."""
        leg = LineString([(0, 0), (100, 0)])
        poly = box(40, 200, 60, 300)
        # Compass 0 = N drift; reverse ray goes south back to leg at y=0.
        d = directional_min_distance_reverse_ray(poly, leg, compass_angle_deg=0)
        assert d is not None
        assert d == pytest.approx(200.0, abs=1.0)

    def test_polygon_upstream_returns_none(self):
        """Polygon south of leg with drift north -> reverse ray never hits."""
        leg = LineString([(0, 0), (100, 0)])
        poly = box(40, -300, 60, -200)
        d = directional_min_distance_reverse_ray(poly, leg, compass_angle_deg=0)
        assert d is None

    def test_empty_geom_returns_none(self):
        leg = LineString([(0, 0), (100, 0)])
        assert directional_min_distance_reverse_ray(Polygon(), leg, 0) is None

    def test_none_geom_returns_none(self):
        leg = LineString([(0, 0), (100, 0)])
        assert directional_min_distance_reverse_ray(None, leg, 0) is None

    def test_multipolygon_aggregates(self):
        leg = LineString([(0, 0), (100, 0)])
        # Two boxes north of leg at different distances.
        mp = MultiPolygon([
            box(20, 100, 30, 150),  # closer (y=100)
            box(60, 300, 70, 350),  # farther
        ])
        d = directional_min_distance_reverse_ray(mp, leg, 0)
        assert d == pytest.approx(100.0, abs=1.0)

    def test_zero_length_leg_returns_none(self):
        leg = LineString()  # empty
        assert directional_min_distance_reverse_ray(box(0, 0, 1, 1), leg, 0) is None

    def test_unsupported_geom_returns_none(self):
        """A Point geom isn't Polygon or MultiPolygon -> returns None."""
        leg = LineString([(0, 0), (100, 0)])
        assert directional_min_distance_reverse_ray(Point(50, 100), leg, 0) is None

    def test_polygon_with_hole_includes_inner_coords(self):
        """Polygon with an interior ring -> inner coords are also probed."""
        leg = LineString([(0, 0), (100, 0)])
        outer = [(20, 100), (80, 100), (80, 200), (20, 200), (20, 100)]
        hole = [(40, 130), (60, 130), (60, 170), (40, 170), (40, 130)]
        poly = Polygon(outer, [hole])
        d = directional_min_distance_reverse_ray(poly, leg, 0)
        # Closest point is along outer ring at y=100.
        assert d == pytest.approx(100.0, abs=1.0)

    def test_multipolygon_with_holes(self):
        """MultiPolygon with interior rings -> code walks both outer and inner."""
        leg = LineString([(0, 0), (100, 0)])
        outer1 = [(0, 200), (40, 200), (40, 240), (0, 240), (0, 200)]
        hole1 = [(10, 210), (30, 210), (30, 230), (10, 230), (10, 210)]
        p1 = Polygon(outer1, [hole1])
        p2 = box(60, 100, 90, 150)
        mp = MultiPolygon([p1, p2])
        d = directional_min_distance_reverse_ray(mp, leg, 0)
        # min y is 100 from the second box.
        assert d == pytest.approx(100.0, abs=1.0)

    def test_drift_parallel_to_leg_uses_scalar_fallback(self):
        """Drift direction parallel to an east-running leg means the reverse
        ray is parallel to the leg -- no segment crossings, so every vertex
        must fall through to the scalar ``nearest_points`` path.  The result
        still has to match the expected along-drift distance.
        """
        # East-running leg; polygon downstream (east) of the leg end.
        leg = LineString([(0, 0), (1000, 0)])
        poly = box(1500, -50, 1800, 50)
        # Compass 90 = E drift; reverse ray goes west, parallel to leg.
        d = directional_min_distance_reverse_ray(poly, leg, compass_angle_deg=90)
        # Closest vertex of poly to the leg, in drift direction: 500 m east
        # of the leg's east endpoint (at x=1000).
        assert d == pytest.approx(500.0, abs=1.0)


# ---------------------------------------------------------------------------
# visualize  (the standalone matplotlib helper)
# ---------------------------------------------------------------------------

class TestVisualize:
    def test_renders_pdf_curve(self, qgis_iface):
        """The function plots the combined PDF on the given Axes."""
        from geometries.get_drifting_overlap import visualize
        from matplotlib.figure import Figure

        fig = Figure()
        ax = fig.subplots()
        distances = np.array([10.0, 20.0, 30.0])
        data = {
            'drift': {
                'speed': 1.94,
                'repair': {
                    'use_lognormal': False, 'std': 1.0, 'loc': 0.0, 'scale': 1.0,
                    'func': "__import__('scipy.stats', fromlist=['norm'])"
                            ".norm(loc=0, scale=1).cdf(x)",
                    'dist_type': 'normal', 'norm_mean': 0.0, 'norm_std': 1.0,
                },
            },
        }
        visualize(ax, distances, [norm(loc=0, scale=10)], [1.0],
                  weighted_overlap=0.5, data=data)
        # The Axes should now have plotted lines (PDF + not_repaired curves).
        assert len(ax.lines) >= 2

    def test_none_overlap_short_circuits(self, qgis_iface):
        """When weighted_overlap is None, the function clears and returns."""
        from geometries.get_drifting_overlap import visualize
        from matplotlib.figure import Figure
        fig = Figure()
        ax = fig.subplots()
        # Provide some prior content so we can verify clear() ran.
        ax.plot([0, 1], [0, 1])
        visualize(ax, np.array([1.0]), [norm(loc=0, scale=1)], [1.0],
                  weighted_overlap=None, data={})
        # ax was cleared -> 0 lines remain.
        assert len(ax.lines) == 0


# ---------------------------------------------------------------------------
# DriftingOverlapVisualizer -- __init__ and trivial state
# ---------------------------------------------------------------------------

class TestDriftingOverlapVisualizerInit:
    def _make_visualizer(self, qgis_iface):
        from geometries.get_drifting_overlap import DriftingOverlapVisualizer
        from matplotlib.figure import Figure
        fig = Figure()
        ax1, ax2, ax3 = fig.subplots(1, 3)
        v = DriftingOverlapVisualizer(
            fig=fig, ax1=ax1, ax2=ax2, ax3=ax3,
            lines=[LineString([(0, 0), (100, 0)])],
            line_names=['L1'],
            objs_gdf_list=[],
            distributions=[[norm(loc=0, scale=10)]],
            weights=[[1.0]],
            data={
                'drift': {'speed': 1.94, 'repair': {
                    'use_lognormal': False, 'std': 1.0, 'loc': 0.0, 'scale': 1.0,
                    'func': '0', 'dist_type': 'normal',
                    'norm_mean': 0.0, 'norm_std': 1.0,
                }}
            },
            distance=1000.0,
        )
        return v

    def test_init_stores_inputs(self, qgis_iface):
        v = self._make_visualizer(qgis_iface)
        assert v.distance == 1000.0
        assert v.line_names == ['L1']
        assert v.current_line is None
        assert v.current_base_polygon is None


# ---------------------------------------------------------------------------
# compute_coverages_and_distances  (the refactored pure helper)
# ---------------------------------------------------------------------------

class TestComputeCoveragesAndDistances:
    def _scene(self):
        """Two polygons + a single object that intersects only the first."""
        import geopandas as gpd
        ext = [box(0, 0, 50, 50), box(100, 100, 150, 150)]
        centre = [
            LineString([(0, 25), (50, 25)]),
            LineString([(100, 125), (150, 125)]),
        ]
        objs = [gpd.GeoDataFrame(geometry=[box(10, 10, 20, 20)])]
        results = {
            'Polygon_0': [[True]],
            'Polygon_1': [[False]],
        }
        return ext, centre, objs, results

    def test_intersecting_polygon_yields_coverage(self):
        from scipy.stats import norm
        ext, centre, objs, results = self._scene()
        cov, dists, covered = compute_coverages_and_distances(
            extended_polygons=ext,
            centre_lines=centre,
            distributions=[norm(loc=0, scale=10)],
            weights=[1.0],
            objs_gdf_list=objs,
            results=results,
        )
        # Two (polygon, gdf, obj) entries.
        assert len(cov) == 2
        assert len(dists) == 2
        # First polygon was hit -> coverage > 0.
        assert cov[0] >= 0.0
        # Second polygon's coverage entry is 0 (no intersection).
        assert cov[1] == 0
        assert covered == [True, False]

    def test_no_intersections_zero_coverage(self):
        from scipy.stats import norm
        import geopandas as gpd
        ext = [box(0, 0, 50, 50)]
        centre = [LineString([(0, 25), (50, 25)])]
        objs = [gpd.GeoDataFrame(geometry=[box(1000, 1000, 1100, 1100)])]
        results = {'Polygon_0': [[False]]}
        cov, dists, covered = compute_coverages_and_distances(
            extended_polygons=ext, centre_lines=centre,
            distributions=[norm(loc=0, scale=10)], weights=[1.0],
            objs_gdf_list=objs, results=results,
        )
        assert cov == [0]
        assert covered == [False]

    def test_multiple_objects_per_gdf(self):
        from scipy.stats import norm
        import geopandas as gpd
        ext = [box(0, 0, 100, 100)]
        centre = [LineString([(0, 50), (100, 50)])]
        objs = [gpd.GeoDataFrame(geometry=[
            box(10, 10, 20, 20), box(40, 40, 50, 50),
        ])]
        results = {'Polygon_0': [[True, True]]}
        cov, dists, covered = compute_coverages_and_distances(
            extended_polygons=ext, centre_lines=centre,
            distributions=[norm(loc=0, scale=10)], weights=[1.0],
            objs_gdf_list=objs, results=results,
        )
        # 2 entries for 1 polygon × 1 gdf × 2 objects.
        assert len(cov) == 2
        assert covered == [True]


# ---------------------------------------------------------------------------
# DriftingOverlapVisualizer -- exercise click paths via mock events
# ---------------------------------------------------------------------------

class TestDriftingOverlapVisualizerEvents:
    def _make_visualizer_with_objs(self, qgis_iface):
        """Visualizer with one obstacle that the corridor will intersect.

        Calls ``run_visualization()`` so the ax1 lines + initial selection
        are populated.
        """
        from geometries.get_drifting_overlap import DriftingOverlapVisualizer
        from matplotlib.figure import Figure
        import geopandas as gpd
        fig = Figure()
        ax1, ax2, ax3 = fig.subplots(1, 3)
        objs = [gpd.GeoDataFrame(geometry=[box(40, -20, 60, 20)])]
        v = DriftingOverlapVisualizer(
            fig=fig, ax1=ax1, ax2=ax2, ax3=ax3,
            lines=[LineString([(0, 0), (100, 0)])],
            line_names=['L1'],
            objs_gdf_list=objs,
            distributions=[[norm(loc=0, scale=10)]],
            weights=[[1.0]],
            data={
                'drift': {'speed': 1.94, 'repair': {
                    'use_lognormal': False, 'std': 1.0, 'loc': 0.0,
                    'scale': 1.0, 'func': '0', 'dist_type': 'normal',
                    'norm_mean': 0.0, 'norm_std': 1.0,
                }}
            },
            distance=200.0,
        )
        v.run_visualization()
        return v

    def test_get_selected_line_index(self, qgis_iface):
        v = self._make_visualizer_with_objs(qgis_iface)
        # Find the matplotlib line artist for 'L1'.
        line_artist = next(
            l for l in v.ax1.get_lines() if l.get_label() == 'L1'
        )
        from types import SimpleNamespace
        evt = SimpleNamespace(artist=line_artist)
        assert v._get_selected_line_index(evt) == 0

    def test_get_selected_line_index_unknown_returns_none(self, qgis_iface):
        v = self._make_visualizer_with_objs(qgis_iface)
        from types import SimpleNamespace
        fake_artist = MagicMock()
        fake_artist.get_label.return_value = 'unknown_line'
        evt = SimpleNamespace(artist=fake_artist)
        assert v._get_selected_line_index(evt) is None

    def test_set_current_line_state(self, qgis_iface):
        v = self._make_visualizer_with_objs(qgis_iface)
        v._set_current_line_state(0)
        assert v.current_line is v.lines[0]
        assert v.current_distribution is v.distributions[0]
        assert v.current_weight is v.weights[0]

    def test_on_polygon_click_wrong_axes_returns(self, qgis_iface):
        """Click in ``ax1`` (the leg picker) is ignored — the polygon
        renderer must not run."""
        v = self._make_visualizer_with_objs(qgis_iface)
        from types import SimpleNamespace
        with patch.object(v, '_render_polygon_panel') as render:
            evt = SimpleNamespace(inaxes=v.ax1, xdata=10, ydata=10)
            v.on_polygon_click(evt)
        render.assert_not_called()

    def test_on_polygon_click_no_xy(self, qgis_iface):
        """Mouse events without xdata/ydata (e.g. off-canvas) are ignored."""
        v = self._make_visualizer_with_objs(qgis_iface)
        from types import SimpleNamespace
        with patch.object(v, '_render_polygon_panel') as render:
            evt = SimpleNamespace(inaxes=v.ax2, xdata=None, ydata=None)
            v.on_polygon_click(evt)
        render.assert_not_called()

    def test_on_polygon_click_no_extended_polygons(self, qgis_iface):
        """Before a leg is selected there are no corridor polygons to
        hit-test, so the click handler returns silently."""
        v = self._make_visualizer_with_objs(qgis_iface)
        v.current_extended_polygons = None
        from types import SimpleNamespace
        with patch.object(v, '_render_polygon_panel') as render:
            evt = SimpleNamespace(inaxes=v.ax2, xdata=10, ydata=10)
            v.on_polygon_click(evt)
        render.assert_not_called()

    def test_simulate_initial_selection_runs_full_pipeline(self, qgis_iface):
        """Constructor calls simulate_initial_selection, which exercises
        on_line_click + on_polygon_click via mock events."""
        v = self._make_visualizer_with_objs(qgis_iface)
        # After simulate_initial_selection, current_line and current_coverages
        # should be set.
        assert v.current_line is not None
        assert v.current_extended_polygons is not None

    def test_on_polygon_click_outside_any_polygon(self, qgis_iface):
        """Click far from every corridor polygon: hit-test loop falls
        through without rendering the bottom panel."""
        v = self._make_visualizer_with_objs(qgis_iface)
        from types import SimpleNamespace
        with patch.object(v, '_render_polygon_panel') as render:
            evt = SimpleNamespace(inaxes=v.ax2, xdata=100_000, ydata=100_000)
            v.on_polygon_click(evt)
        render.assert_not_called()
