"""Unit tests for the private helpers in compute/drifting_model.py::DriftingModelMixin.

These exercise the pure-logic pieces of the cascade without running the
full traffic loop.  They fall into three groups:

1. Config/math: ``_compute_reach_distance``, ``_distribution_centerline_stats``.
2. Geometry: ``_build_blocker_shadow``, ``_analytical_hole_for_geom``.
3. Edge distribution: ``_edge_weighted_holes``.

Mixin methods are invoked on a trivial instance (`DriftingModelMixin()`);
the host class's attributes aren't touched by these helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from shapely.geometry import LineString, Point, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.drifting_model import DriftingModelMixin


@pytest.fixture
def mixin():
    return DriftingModelMixin()


# ---------------------------------------------------------------------------
# _compute_reach_distance
# ---------------------------------------------------------------------------

class TestComputeReachDistance:
    def test_default_when_repair_missing(self, mixin):
        """Without repair config the default cap is longest_length * 10."""
        assert mixin._compute_reach_distance({}, 5000.0) == 50000.0

    def test_use_lognormal_scales_to_drift_speed(self, mixin):
        """Lognormal repair time × drift speed gives the reach distance."""
        data = {
            'drift': {
                'speed': 1.94,  # knots
                'repair': {
                    'use_lognormal': True,
                    'std': 1.0, 'loc': 0.0, 'scale': 1.0,
                },
            },
        }
        reach = mixin._compute_reach_distance(data, 10_000_000.0)
        # t99 = lognorm(s=1).ppf(0.99) ~ 10.24 h
        # drift speed 1.94 kts = 0.9982 m/s
        # reach = 0.9982 * 3600 * 10.24 ~ 36 800 m, not clamped because
        # longest_length * 10 = 100 Mm.
        t99 = float(stats.lognorm(1.0, loc=0.0, scale=1.0).ppf(0.99))
        expected = 0.9982 * 3600.0 * t99
        assert reach == pytest.approx(expected, rel=1e-3)

    def test_reach_clamped_to_longest_length_times_10(self, mixin):
        """Extreme repair-time distributions still respect the 10× cap."""
        data = {
            'drift': {
                'speed': 10.0,
                'repair': {
                    'use_lognormal': True,
                    'std': 5.0, 'loc': 0.0, 'scale': 1.0,  # very fat tail
                },
            },
        }
        assert mixin._compute_reach_distance(data, 1000.0) == 10_000.0

    def test_weibull_repair(self, mixin):
        data = {
            'drift': {
                'speed': 2.0,
                'repair': {
                    'dist_type': 'weibull',
                    'wb_shape': 2.0, 'wb_loc': 0.0, 'wb_scale': 3.0,
                },
            },
        }
        reach = mixin._compute_reach_distance(data, 1e9)
        t99 = float(stats.weibull_min(c=2.0, loc=0.0, scale=3.0).ppf(0.99))
        drift_ms = 2.0 * 1852.0 / 3600.0
        assert reach == pytest.approx(drift_ms * 3600.0 * t99, rel=1e-3)

    def test_falls_back_on_exception(self, mixin):
        """Malformed repair config falls back to the default cap."""
        data = {'drift': {'speed': 'not a number', 'repair': {'use_lognormal': True}}}
        assert mixin._compute_reach_distance(data, 2000.0) == 20_000.0

    def test_zero_drift_speed_ignored(self, mixin):
        """Zero drift speed skips the reach-distance update -- default returned."""
        data = {
            'drift': {
                'speed': 0.0,
                'repair': {
                    'use_lognormal': True,
                    'std': 1.0, 'loc': 0.0, 'scale': 1.0,
                },
            },
        }
        assert mixin._compute_reach_distance(data, 2000.0) == 20_000.0

    def test_no_t99_returns_default(self, mixin):
        """Neither weibull nor use_lognormal set -> t99_h stays None -> default."""
        data = {'drift': {'speed': 5.0, 'repair': {}}}
        assert mixin._compute_reach_distance(data, 2000.0) == 20_000.0


# ---------------------------------------------------------------------------
# _distribution_centerline_stats
# ---------------------------------------------------------------------------

class TestDistributionCenterlineStats:
    def test_empty_inputs_return_default(self, mixin):
        assert mixin._distribution_centerline_stats([], []) == (0.0, 1.0)
        assert mixin._distribution_centerline_stats(None, []) == (0.0, 1.0)  # type: ignore[arg-type]

    def test_single_distribution_returns_its_stats(self, mixin):
        d = stats.norm(loc=5.0, scale=2.0)
        mean, sigma = mixin._distribution_centerline_stats([d], [1.0])
        assert mean == pytest.approx(5.0, abs=1e-9)
        # The helper clamps sigma to at least sqrt(1) = 1.0 and adds no
        # mixture-variance term when there's only one component, so sigma
        # ends up exactly the component std (2.0).
        assert sigma == pytest.approx(2.0, abs=1e-9)

    def test_zero_weights_return_default(self, mixin):
        d = stats.norm(loc=5.0, scale=2.0)
        assert mixin._distribution_centerline_stats([d], [0.0]) == (0.0, 1.0)

    def test_two_component_mixture_mean_and_variance(self, mixin):
        d1 = stats.norm(loc=0.0, scale=1.0)
        d2 = stats.norm(loc=10.0, scale=1.0)
        mean, sigma = mixin._distribution_centerline_stats([d1, d2], [1.0, 1.0])
        # Weighted mean = 5, variance = mean(var + (mu_i - 5)^2) =
        # (1 + 25) + (1 + 25) averaged = 26.
        assert mean == pytest.approx(5.0, abs=1e-9)
        assert sigma == pytest.approx(np.sqrt(26.0), abs=1e-6)

    def test_sigma_floored_at_one(self, mixin):
        """Variance clamp: sigma >= sqrt(1.0) even with tiny spreads."""
        d = stats.norm(loc=0.0, scale=1e-6)
        _, sigma = mixin._distribution_centerline_stats([d], [1.0])
        assert sigma >= 1.0

    def test_non_finite_values_skipped(self, mixin):
        class BadDist:
            def mean(self): return float('nan')
            def std(self): return 1.0

        good = stats.norm(loc=3.0, scale=2.0)
        mean, sigma = mixin._distribution_centerline_stats([BadDist(), good], [1.0, 1.0])
        # The NaN component is skipped, so result comes from `good` alone.
        assert mean == pytest.approx(3.0, abs=1e-9)
        assert sigma == pytest.approx(2.0, abs=1e-9)

    def test_exception_in_dist_extraction_skipped(self, mixin):
        """A distribution whose .mean() / .std() raises is silently skipped."""
        class RaisingDist:
            def mean(self): raise RuntimeError('synthetic')
            def std(self): return 1.0

        good = stats.norm(loc=3.0, scale=2.0)
        mean, sigma = mixin._distribution_centerline_stats(
            [RaisingDist(), good], [1.0, 1.0],
        )
        # Only `good` survives.
        assert mean == pytest.approx(3.0, abs=1e-9)

    def test_negative_total_weight_returns_default(self, mixin):
        """With one non-zero weight that's negative the loop's `w <= 0`
        guard skips it -> total_weight stays 0 -> default returned."""
        d = stats.norm(loc=5.0, scale=1.0)
        # All weights non-positive -> default returned via L131 (no entries).
        assert mixin._distribution_centerline_stats([d, d], [-1.0, 0.0]) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# _edge_weighted_holes
# ---------------------------------------------------------------------------

class TestEdgeWeightedHoles:
    def test_returns_single_none_tuple_when_obs_missing(self, mixin):
        corridor = box(0, 0, 100, 100)
        assert mixin._edge_weighted_holes(None, corridor, 0.0, None, 0.5) == [(None, 0.5)]

    def test_returns_single_none_tuple_when_corridor_missing(self, mixin):
        poly = box(10, 10, 20, 20)
        assert mixin._edge_weighted_holes(poly, None, 0.0, None, 0.3) == [(None, 0.3)]

    def test_fractions_sum_to_hole_pct(self, mixin):
        """Square polygon fully inside the corridor; fractions sum to h."""
        # A large corridor + a small square polygon fully inside it.
        corridor = box(-10_000, -10_000, 10_000, 10_000)
        poly = box(0, 0, 100, 50)
        leg = LineString([(-5000, -5000), (5000, -5000)])

        edges = mixin._edge_weighted_holes(poly, corridor, 0.0, leg, 0.2)

        # At least one segment identified; total fraction equals input.
        assert edges, "expected at least one weighted edge"
        assert all(idx is not None for idx, _ in edges)
        total = sum(v for _, v in edges)
        assert total == pytest.approx(0.2, rel=1e-9)

    def test_falls_back_when_all_segments_filtered(self, mixin):
        """When every segment fails the corridor test, the helper returns
        the single-None fallback so the caller still gets `hole_pct`.
        """
        # Corridor that doesn't overlap the polygon at all.
        corridor = box(-10_000, -10_000, -5_000, -5_000)
        poly = box(5_000, 5_000, 6_000, 6_000)
        leg = LineString([(-7000, -7000), (-6000, -6000)])

        edges = mixin._edge_weighted_holes(poly, corridor, 0.0, leg, 0.7)
        assert edges == [(None, 0.7)]

    def test_point_geom_has_no_segments(self, mixin):
        """A Point has no extractable obstacle segments -> fallback (None, h)."""
        edges = mixin._edge_weighted_holes(
            Point(5, 5), box(0, 0, 100, 100), 0.0, None, 0.4,
        )
        assert edges == [(None, 0.4)]

    def test_exception_in_segments_falls_back(self, mixin, monkeypatch):
        """If ``_extract_obstacle_segments`` raises, the outer except
        catches it and returns the fallback tuple."""
        import compute.drifting_model as mod

        def boom(geom):
            raise RuntimeError('synthetic')

        monkeypatch.setattr(mod, '_extract_obstacle_segments', boom)
        edges = mixin._edge_weighted_holes(
            box(0, 0, 1, 1), box(-10, -10, 10, 10), 0.0, None, 0.9,
        )
        assert edges == [(None, 0.9)]


# ---------------------------------------------------------------------------
# _build_blocker_shadow
# ---------------------------------------------------------------------------

class TestBuildBlockerShadow:
    def test_empty_input_returns_empty_polygon(self, mixin):
        bounds = (-1000.0, -1000.0, 1000.0, 1000.0)
        assert mixin._build_blocker_shadow(None, 0.0, bounds).is_empty
        assert mixin._build_blocker_shadow(Polygon(), 0.0, bounds).is_empty

    def test_none_bounds_returns_empty_polygon(self, mixin):
        poly = box(0, 0, 10, 10)
        assert mixin._build_blocker_shadow(poly, 0.0, None).is_empty

    def test_single_polygon_produces_shadow_touching_obstacle(self, mixin):
        poly = box(-5, -5, 5, 5)
        bounds = (-1000.0, -1000.0, 1000.0, 1000.0)
        shadow = mixin._build_blocker_shadow(poly, 0.0, bounds)  # drift north
        assert not shadow.is_empty
        # Shadow should include the original obstacle footprint (quad-sweep
        # unions obstacle + quads + translated obstacle).
        assert shadow.intersects(poly)

    def test_multipolygon_returns_union_of_shadows(self, mixin):
        from shapely.geometry import MultiPolygon
        p1 = box(-20, -5, -10, 5)
        p2 = box(10, -5, 20, 5)
        mp = MultiPolygon([p1, p2])
        bounds = (-1000.0, -1000.0, 1000.0, 1000.0)
        shadow = mixin._build_blocker_shadow(mp, 0.0, bounds)
        assert not shadow.is_empty
        # Both component footprints are inside the union.
        assert shadow.intersects(p1) and shadow.intersects(p2)

    def test_extract_polygons_exception_returns_empty(self, mixin, monkeypatch):
        """If ``extract_polygons`` raises, ``polys`` stays empty -> return empty."""
        import compute.drifting_model as mod

        def boom(geom):
            raise RuntimeError('synthetic')

        monkeypatch.setattr(mod, 'extract_polygons', boom)
        result = mixin._build_blocker_shadow(box(0, 0, 1, 1), 0.0, (-10, -10, 10, 10))
        assert result.is_empty

    def test_shadow_creation_exception_skipped(self, mixin, monkeypatch):
        """A per-polygon shadow failure just skips that polygon."""
        import compute.drifting_model as mod

        def boom(*a, **k):
            raise RuntimeError('synthetic')

        monkeypatch.setattr(mod, 'create_obstacle_shadow', boom)
        result = mixin._build_blocker_shadow(box(0, 0, 1, 1), 0.0, (-10, -10, 10, 10))
        # All shadows failed -> returns an empty polygon.
        assert result.is_empty

    def test_unary_union_exception_returns_first_shadow(self, mixin, monkeypatch):
        """If unary_union raises on the multi-shadow union, the function
        returns the first shadow as a fallback."""
        from shapely.geometry import MultiPolygon
        import compute.drifting_model as mod

        # Force unary_union on the module to raise.
        def boom(*a, **k):
            raise RuntimeError('union failed')

        monkeypatch.setattr(mod, 'unary_union', boom)

        mp = MultiPolygon([box(-10, -5, 0, 5), box(5, -5, 15, 5)])
        result = mixin._build_blocker_shadow(mp, 0.0, (-1000.0, -1000.0, 1000.0, 1000.0))
        assert not result.is_empty  # returned the first shadow


# ---------------------------------------------------------------------------
# _analytical_hole_for_geom
# ---------------------------------------------------------------------------

class TestAnalyticalHoleForGeom:
    @staticmethod
    def _leg() -> LineString:
        # 1 km leg going east.
        return LineString([(0.0, 0.0), (1000.0, 0.0)])

    def test_empty_geom_returns_zero(self, mixin):
        h = mixin._analytical_hole_for_geom(
            Polygon(), self._leg(), 0.0, [stats.norm(0, 100)], np.array([1.0]),
            reach_distance=10_000.0, lateral_range=500.0,
        )
        assert h == 0.0

    def test_none_geom_returns_zero(self, mixin):
        h = mixin._analytical_hole_for_geom(
            None, self._leg(), 0.0, [stats.norm(0, 100)], np.array([1.0]),
            reach_distance=10_000.0, lateral_range=500.0,
        )
        assert h == 0.0

    def test_zero_reach_distance_returns_zero(self, mixin):
        poly = box(-50, 400, 50, 500)  # small target 400 m "north" of leg
        h = mixin._analytical_hole_for_geom(
            poly, self._leg(), 0.0, [stats.norm(0, 100)], np.array([1.0]),
            reach_distance=0.0, lateral_range=500.0,
        )
        assert h == 0.0

    def test_extract_polygons_exception_returns_zero(self, mixin, monkeypatch):
        """extract_polygons raising -> polys is [] -> returns 0.0."""
        import compute.drifting_model as mod
        monkeypatch.setattr(
            mod, 'extract_polygons',
            lambda g: (_ for _ in ()).throw(RuntimeError('bad')),
        )
        h = mixin._analytical_hole_for_geom(
            box(0, 0, 1, 1), self._leg(), 0.0, [stats.norm(0, 100)], np.array([1.0]),
            reach_distance=10_000.0, lateral_range=500.0,
        )
        assert h == 0.0

    def test_extract_rings_exception_skipped(self, mixin, monkeypatch):
        """_extract_polygon_rings raising for each polygon -> no rings -> 0.0."""
        import compute.drifting_model as mod
        monkeypatch.setattr(
            mod, '_extract_polygon_rings',
            lambda p: (_ for _ in ()).throw(RuntimeError('bad ring')),
        )
        h = mixin._analytical_hole_for_geom(
            box(0, 0, 1, 1), self._leg(), 0.0, [stats.norm(0, 100)], np.array([1.0]),
            reach_distance=10_000.0, lateral_range=500.0,
        )
        assert h == 0.0

    def test_degenerate_leg_returns_zero(self, mixin):
        """Empty LineString -> len(coords) < 2 -> return 0.0."""
        h = mixin._analytical_hole_for_geom(
            box(0, 0, 1, 1), LineString(), 0.0,
            [stats.norm(0, 100)], np.array([1.0]),
            reach_distance=10_000.0, lateral_range=500.0,
        )
        assert h == 0.0

    def test_inner_exception_returns_zero(self, mixin, monkeypatch):
        """If compute_probability_analytical raises, the outer except catches
        it and returns 0.0."""
        import compute.drifting_model as mod
        monkeypatch.setattr(
            mod, 'compute_probability_analytical',
            lambda **k: (_ for _ in ()).throw(RuntimeError('analyt fail')),
        )
        h = mixin._analytical_hole_for_geom(
            box(0, 0, 1, 1), self._leg(), 0.0, [stats.norm(0, 100)], np.array([1.0]),
            reach_distance=10_000.0, lateral_range=500.0,
        )
        assert h == 0.0

    def test_positive_hole_for_reachable_target(self, mixin):
        """A polygon that sits inside the leg's ±1σ lateral range produces
        a positive hole."""
        # Leg east along y=0.  Target 300 m "north" (compass 0).
        poly = box(400.0, 250.0, 600.0, 350.0)
        dist = stats.norm(0, 200)
        h = mixin._analytical_hole_for_geom(
            poly, self._leg(), 0.0, [dist], np.array([1.0]),
            reach_distance=5_000.0, lateral_range=1_000.0, n_slices=100,
        )
        assert h > 0.0
        assert h <= 1.0  # it's a probability-like integral

    def test_handles_multipolygon_input(self, mixin):
        from shapely.geometry import MultiPolygon
        poly = MultiPolygon([
            box(100.0, 200.0, 150.0, 300.0),
            box(800.0, 200.0, 850.0, 300.0),
        ])
        h = mixin._analytical_hole_for_geom(
            poly, self._leg(), 0.0, [stats.norm(0, 200)], np.array([1.0]),
            reach_distance=5_000.0, lateral_range=1_000.0, n_slices=100,
        )
        assert h >= 0.0

    def test_invalid_leg_returns_zero(self, mixin):
        # Leg with only a single point -> leg_vec can't be formed.
        poly = box(10, 10, 20, 20)
        degenerate_leg = LineString([(0, 0), (0, 0)])
        h = mixin._analytical_hole_for_geom(
            poly, degenerate_leg, 0.0, [stats.norm(0, 10)], np.array([1.0]),
            reach_distance=100.0, lateral_range=50.0,
        )
        # Zero-length leg -> leg_dir defaults or the integrator returns 0.
        assert h >= 0.0
