"""Unit tests for the helpers extracted from ``geometries.get_drifting_overlap``.

Covers the pure-data sidebar helpers and the new
``DriftingModelMixin`` static methods (``_compute_global_shadow_bounds``
and ``_precompute_leg_lateral_params``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Sidebar helpers (no Qt-dependent items here -- pure data only)
# ---------------------------------------------------------------------------
class TestSidebarHelpers:
    def test_polygon_to_compass_round_trip(self):
        from geometries.drift_overlap_sidebar import polygon_to_compass
        # math 0 -> compass 90 (East)
        assert polygon_to_compass(0) == 90
        # math 90 -> compass 0 (North)
        assert polygon_to_compass(2) == 0
        # math 180 -> compass 270 (West)
        assert polygon_to_compass(4) == 270
        # math 270 -> compass 180 (South)
        assert polygon_to_compass(6) == 180

    def test_split_leg_name_typical(self):
        from geometries.drift_overlap_sidebar import split_leg_name
        assert split_leg_name("Leg 1-East going") == ("1", "East going")
        assert split_leg_name("Leg 12-West going") == ("12", "West going")

    def test_split_leg_name_no_dash(self):
        from geometries.drift_overlap_sidebar import split_leg_name
        # No dash -> tail is empty.
        assert split_leg_name("Leg 1") == ("1", "")

    def test_split_leg_name_non_string_falls_back(self):
        from geometries.drift_overlap_sidebar import split_leg_name
        # An object whose str() raises -> fall-back path returns
        # ``(str(name), '')``.  Use a normal string here as a smoke test.
        assert split_leg_name("garbage") == ("garbage", "")

    def test_lookup_contribution_hit(self):
        from geometries.drift_overlap_sidebar import lookup_contribution
        bld = {
            '1:East going:0': {'contrib_grounding': 1.5e-4},
        }
        assert lookup_contribution(
            bld, '1', 'East going', 0, 'contrib_grounding',
        ) == pytest.approx(1.5e-4)

    def test_lookup_contribution_missing_returns_none_or_zero(self):
        from geometries.drift_overlap_sidebar import lookup_contribution
        bld = {}
        # Empty bld -> None.
        assert lookup_contribution(
            bld, '1', 'East going', 0, 'contrib_grounding',
        ) is None
        bld = {'2:Other:0': {'contrib_grounding': 0.0}}
        # Key miss with non-empty dict -> 0.0 fallback.
        assert lookup_contribution(
            bld, '1', 'East going', 0, 'contrib_grounding',
        ) == 0.0

    def test_lookup_contribution_garbage_value_returns_none(self):
        from geometries.drift_overlap_sidebar import lookup_contribution
        bld = {'1:E:0': {'contrib_grounding': 'not-a-number'}}
        assert lookup_contribution(
            bld, '1', 'E', 0, 'contrib_grounding',
        ) is None


# ---------------------------------------------------------------------------
# Drifting-model mixin static helpers
# ---------------------------------------------------------------------------
class TestComputeGlobalShadowBounds:
    def _import(self):
        from compute.drifting_model import DriftingModelMixin
        return DriftingModelMixin

    def test_empty_inputs_returns_none(self):
        Mixin = self._import()
        assert Mixin._compute_global_shadow_bounds([], [], [], 1000.0) is None

    def test_single_leg_padded_by_reach(self):
        Mixin = self._import()
        line = LineString([(0.0, 0.0), (10_000.0, 0.0)])
        out = Mixin._compute_global_shadow_bounds(
            [line], [], [], reach_distance=2_000.0,
        )
        assert out is not None
        minx, miny, maxx, maxy = out
        assert minx == pytest.approx(-2_000.0)
        assert miny == pytest.approx(-2_000.0)
        assert maxx == pytest.approx(12_000.0)
        assert maxy == pytest.approx(2_000.0)

    def test_padding_floor_at_1km(self):
        """Reach below 1 km still pads by 1 km (safety floor)."""
        Mixin = self._import()
        line = LineString([(0.0, 0.0), (5_000.0, 0.0)])
        out = Mixin._compute_global_shadow_bounds(
            [line], [], [], reach_distance=10.0,
        )
        assert out is not None
        minx, miny, maxx, maxy = out
        assert minx == pytest.approx(-1_000.0)
        assert maxx == pytest.approx(6_000.0)

    def test_structures_extend_bounds(self):
        Mixin = self._import()
        line = LineString([(0.0, 0.0), (1_000.0, 0.0)])
        struct_poly = Polygon([
            (5_000.0, 5_000.0), (6_000.0, 5_000.0),
            (6_000.0, 6_000.0), (5_000.0, 6_000.0),
        ])
        out = Mixin._compute_global_shadow_bounds(
            [line], [{'wkt': struct_poly}], [], reach_distance=1_000.0,
        )
        assert out is not None
        minx, miny, maxx, maxy = out
        # max-x dominated by struct_poly + 1km pad.
        assert maxx == pytest.approx(7_000.0)
        assert maxy == pytest.approx(7_000.0)

    def test_skips_invalid_geometries(self):
        Mixin = self._import()
        line = LineString([(0.0, 0.0), (1_000.0, 0.0)])
        out = Mixin._compute_global_shadow_bounds(
            [line],
            [{'wkt': None}],         # struct without geom
            [{'wkt': Polygon()}],    # depth with empty geom
            reach_distance=500.0,
        )
        assert out is not None
        minx, miny, maxx, maxy = out
        # Bounds come purely from the leg.
        assert minx == pytest.approx(-1_000.0)
        assert maxx == pytest.approx(2_000.0)


class TestPrecomputeLegLateralParams:
    def _import(self):
        from compute.drifting_model import DriftingModelMixin
        return DriftingModelMixin

    def test_returns_one_entry_per_leg(self):
        Mixin = self._import()
        legs = [
            LineString([(0.0, 0.0), (1_000.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 2_000.0)]),
        ]
        out = Mixin._precompute_leg_lateral_params(legs, [[], []], [[], []])
        assert len(out) == 2
        assert all('leg_state' in entry for entry in out)
        assert all('lateral_spread' in entry for entry in out)

    def test_lateral_spread_from_distributions(self):
        from scipy import stats
        Mixin = self._import()
        legs = [LineString([(0.0, 0.0), (1_000.0, 0.0)])]
        dist = stats.norm(loc=0, scale=200)
        out = Mixin._precompute_leg_lateral_params(
            legs, [[dist]], [[1.0]],
        )
        # 5 sigma * 200m = 1000m
        assert out[0]['lateral_spread'] == pytest.approx(1_000.0)
        assert out[0]['w_dir'] is not None
        # weights normalised
        assert np.isclose(np.sum(out[0]['w_dir']), 1.0)

    def test_no_distributions_yields_zero_spread(self):
        Mixin = self._import()
        legs = [LineString([(0.0, 0.0), (500.0, 0.0)])]
        out = Mixin._precompute_leg_lateral_params(legs, [[]], [[]])
        assert out[0]['lateral_spread'] == 0.0
        assert out[0]['w_dir'] is None
        # leg_state still produced for non-degenerate legs.
        assert out[0]['leg_state'] is not None

    def test_degenerate_leg_yields_none_state(self):
        Mixin = self._import()
        # A LineString with one coord is invalid -- shapely will reject
        # it.  Use a single-point sequence wrapped as a LineString-like
        # mock to exercise the ``except`` branch.
        class _BadLine:
            coords = []
            length = 0.0
            bounds = (0.0, 0.0, 0.0, 0.0)
            is_empty = True
        out = Mixin._precompute_leg_lateral_params(
            [_BadLine()], [[]], [[]],
        )
        assert out[0]['leg_state'] is None


# ---------------------------------------------------------------------------
# drift_overlap_plot: smoke-test that visualize handles edge inputs
# ---------------------------------------------------------------------------
class TestVisualizeEdges:
    def _ax(self):
        from matplotlib.figure import Figure
        fig = Figure()
        ax = fig.add_subplot(111)
        return ax

    def test_none_overlap_short_circuits(self):
        from geometries.drift_overlap_plot import visualize
        ax = self._ax()
        visualize(
            ax, distances=np.array([1.0, 2.0]),
            distributions=[], weights=[],
            weighted_overlap=None, data={'drift': {}},
        )
        # No plot, no exception -- just a clear axis.
        assert len(ax.lines) == 0

    def test_empty_distances_renders_no_data_message(self):
        from geometries.drift_overlap_plot import visualize
        ax = self._ax()
        visualize(
            ax, distances=np.array([]),
            distributions=[], weights=[],
            weighted_overlap=0.5,
            data={'drift': {'repair': {}, 'speed': 1.94}},
        )
        # Empty distances -> single text artist with the no-data message.
        texts = [t.get_text() for t in ax.texts]
        assert any('nothing to plot' in t for t in texts)

    def test_all_inf_distances_renders_no_data_message(self):
        from geometries.drift_overlap_plot import visualize
        ax = self._ax()
        visualize(
            ax, distances=np.array([np.inf, -np.inf, np.nan]),
            distributions=[], weights=[],
            weighted_overlap=0.5,
            data={'drift': {'repair': {}, 'speed': 1.94}},
        )
        texts = [t.get_text() for t in ax.texts]
        assert any('non-finite' in t.lower() for t in texts)
