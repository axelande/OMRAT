"""Unit tests for geometries/analytical_probability.py.

Focuses on the public ``compute_probability_analytical`` and its small
helpers ``_extract_polygon_rings`` and ``_merge_intervals_vectorized``.
The deeper ``_vectorized_edge_y_intervals`` is exercised via the public
function; its internals would need a full geometric setup to test in
isolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from shapely.geometry import MultiPolygon, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.analytical_probability import (
    _extract_polygon_rings,
    _merge_intervals_vectorized,
    compute_probability_analytical,
)


# ---------------------------------------------------------------------------
# _extract_polygon_rings
# ---------------------------------------------------------------------------

class TestExtractPolygonRings:
    def test_single_polygon_one_ring(self):
        p = box(0, 0, 1, 1)
        rings = _extract_polygon_rings(p)
        assert len(rings) == 1 and rings[0].shape[1] == 2

    def test_polygon_with_hole_two_rings(self):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        p = Polygon(outer, [hole])
        rings = _extract_polygon_rings(p)
        assert len(rings) == 2

    def test_multipolygon_extracts_each_component(self):
        mp = MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)])
        rings = _extract_polygon_rings(mp)
        assert len(rings) == 2

    def test_empty_polygon_yields_empty(self):
        assert _extract_polygon_rings(Polygon()) == []


# ---------------------------------------------------------------------------
# _merge_intervals_vectorized
# ---------------------------------------------------------------------------

class TestMergeIntervalsVectorized:
    def test_returns_empty_when_nothing_valid(self):
        lo = np.array([0.0, 1.0])
        hi = np.array([2.0, 3.0])
        valid = np.array([False, False])
        assert _merge_intervals_vectorized(lo, hi, valid) == []

    def test_single_interval_passthrough(self):
        lo = np.array([1.0])
        hi = np.array([2.0])
        valid = np.array([True])
        assert _merge_intervals_vectorized(lo, hi, valid) == [(1.0, 2.0)]

    def test_overlapping_intervals_merged(self):
        lo = np.array([1.0, 1.5, 5.0])
        hi = np.array([2.0, 3.0, 6.0])
        valid = np.array([True, True, True])
        out = _merge_intervals_vectorized(lo, hi, valid)
        assert out == [(1.0, 3.0), (5.0, 6.0)]

    def test_unsorted_intervals_sorted(self):
        lo = np.array([5.0, 1.0])
        hi = np.array([6.0, 2.0])
        valid = np.array([True, True])
        out = _merge_intervals_vectorized(lo, hi, valid)
        assert out[0][0] < out[1][0]


# ---------------------------------------------------------------------------
# compute_probability_analytical
# ---------------------------------------------------------------------------

class TestComputeProbabilityAnalytical:
    @staticmethod
    def _straight_east_leg():
        return (
            np.array([0.0, 0.0]),          # leg_start
            np.array([1000.0, 0.0]),       # leg_vec (1km east)
            np.array([0.0, 1.0]),          # perp_dir (north)
            np.array([0.0, 1.0]),          # drift_vec (north)
        )

    def test_empty_polygon_rings_return_zero(self):
        ls, lv, pd, dv = self._straight_east_leg()
        assert compute_probability_analytical(
            leg_start=ls, leg_vec=lv, perp_dir=pd, drift_vec=dv,
            distance=1000.0, lateral_range=500.0,
            polygon_rings=[], dists=[stats.norm(0, 100)],
            weights=np.array([1.0]),
        ) == 0.0

    def test_degenerate_ring_skipped(self):
        """Rings with fewer than 3 vertices are ignored."""
        ls, lv, pd, dv = self._straight_east_leg()
        rings = [np.array([[0, 0], [1, 1]])]  # only 2 points
        assert compute_probability_analytical(
            leg_start=ls, leg_vec=lv, perp_dir=pd, drift_vec=dv,
            distance=1000.0, lateral_range=500.0,
            polygon_rings=rings, dists=[stats.norm(0, 100)],
            weights=np.array([1.0]),
        ) == 0.0

    def test_probability_in_zero_one_range(self):
        """A polygon near the leg produces a probability in [0, 1]."""
        ls, lv, pd, dv = self._straight_east_leg()
        poly = box(400.0, 200.0, 600.0, 400.0)
        rings = _extract_polygon_rings(poly)
        p = compute_probability_analytical(
            leg_start=ls, leg_vec=lv, perp_dir=pd, drift_vec=dv,
            distance=1000.0, lateral_range=500.0,
            polygon_rings=rings, dists=[stats.norm(0, 200)],
            weights=np.array([1.0]), n_slices=50,
        )
        assert 0.0 <= p <= 1.0
        assert p > 0.0  # polygon is in reach

    def test_probability_zero_when_target_out_of_lateral_range(self):
        """Polygon far outside lateral_range + drift distance -> near 0."""
        ls, lv, pd, dv = self._straight_east_leg()
        # Polygon 100 km north; lateral range + drift only reach ~1.5km.
        poly = box(0, 100_000, 1000, 101_000)
        rings = _extract_polygon_rings(poly)
        p = compute_probability_analytical(
            leg_start=ls, leg_vec=lv, perp_dir=pd, drift_vec=dv,
            distance=1000.0, lateral_range=500.0,
            polygon_rings=rings, dists=[stats.norm(0, 100)],
            weights=np.array([1.0]), n_slices=50,
        )
        assert p == pytest.approx(0.0, abs=1e-8)

    def test_larger_polygon_yields_more_probability(self):
        ls, lv, pd, dv = self._straight_east_leg()
        small = _extract_polygon_rings(box(400, 200, 410, 210))
        large = _extract_polygon_rings(box(400, 200, 600, 400))
        p_small = compute_probability_analytical(
            leg_start=ls, leg_vec=lv, perp_dir=pd, drift_vec=dv,
            distance=1000.0, lateral_range=500.0,
            polygon_rings=small, dists=[stats.norm(0, 200)],
            weights=np.array([1.0]), n_slices=50,
        )
        p_large = compute_probability_analytical(
            leg_start=ls, leg_vec=lv, perp_dir=pd, drift_vec=dv,
            distance=1000.0, lateral_range=500.0,
            polygon_rings=large, dists=[stats.norm(0, 200)],
            weights=np.array([1.0]), n_slices=50,
        )
        assert p_large > p_small

    def test_multiple_distributions_with_weights(self):
        """A two-component mixture gives a probability between the two
        component-only probabilities when weights match the mixture."""
        ls, lv, pd, dv = self._straight_east_leg()
        rings = _extract_polygon_rings(box(400, 200, 600, 300))
        p_tight = compute_probability_analytical(
            leg_start=ls, leg_vec=lv, perp_dir=pd, drift_vec=dv,
            distance=1000.0, lateral_range=500.0,
            polygon_rings=rings, dists=[stats.norm(0, 100)],
            weights=np.array([1.0]), n_slices=50,
        )
        p_wide = compute_probability_analytical(
            leg_start=ls, leg_vec=lv, perp_dir=pd, drift_vec=dv,
            distance=1000.0, lateral_range=500.0,
            polygon_rings=rings, dists=[stats.norm(0, 400)],
            weights=np.array([1.0]), n_slices=50,
        )
        p_mix = compute_probability_analytical(
            leg_start=ls, leg_vec=lv, perp_dir=pd, drift_vec=dv,
            distance=1000.0, lateral_range=500.0,
            polygon_rings=rings,
            dists=[stats.norm(0, 100), stats.norm(0, 400)],
            weights=np.array([0.5, 0.5]), n_slices=50,
        )
        # Mixture weight is (p_tight + p_wide)/2 (0.5/0.5 combo).
        assert p_mix == pytest.approx((p_tight + p_wide) / 2.0, rel=1e-6)
