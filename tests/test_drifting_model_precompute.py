"""Tests for the precompute + cancellation paths in ``DriftingModelMixin``.

These target specific branches inside ``_precompute_shadow_layer``,
``_precompute_bucket_memo``, ``_precompute_spatial`` and the top-level
``run_drifting_model`` (cancellation handling, __cancelled__ short-circuits,
result-layer error swallowing).  Keeps the existing minimal synthetic
project fixture from ``test_cascade_minimal`` reusable here.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.drifting_model import DriftingModelMixin


@pytest.fixture
def mixin_with_progress():
    """A mixin instance with a progress callback that always returns True."""
    m = DriftingModelMixin()
    m._report_progress = MagicMock(return_value=True)
    m._cascade_bucket_memo = {}
    return m


# ---------------------------------------------------------------------------
# _precompute_shadow_layer direct invocation
# ---------------------------------------------------------------------------

class TestPrecomputeShadowLayer:
    def test_empty_legs_returns_empty_cache(self, mixin_with_progress):
        cache = mixin_with_progress._precompute_shadow_layer(
            transformed_lines=[],
            distributions=[],
            weights=[],
            structures=[],
            depths=[],
            struct_min_dists=None,
            depth_min_dists=None,
            reach_distance=0.0,
            drift_repair={'use_lognormal': False, 'std': 1.0, 'loc': 0.0, 'scale': 1.0},
            drift_speed=1.0,
            use_leg_offset_for_distance=False,
        )
        # No legs -> no cache entries.
        assert '__cancelled__' not in cache
        assert cache == {}

    def test_single_leg_populates_8_directions(self, mixin_with_progress):
        line = LineString([(0, 0), (10_000, 0)])
        struct = {'id': 's1', 'wkt': box(3_000, 500, 3_500, 1_000), 'height': 20.0}
        depth = {'id': 'd1', 'wkt': box(5_000, -1_000, 5_500, -500), 'depth': 5.0}
        from scipy.stats import norm
        dists = [[norm(loc=0, scale=100.0)]]
        wgts = [[1.0]]

        cache = mixin_with_progress._precompute_shadow_layer(
            transformed_lines=[line],
            distributions=dists,
            weights=wgts,
            structures=[struct],
            depths=[depth],
            struct_min_dists=None,
            depth_min_dists=None,
            reach_distance=5_000.0,
            drift_repair={'use_lognormal': False, 'std': 1.0, 'loc': 0.0, 'scale': 1.0},
            drift_speed=1.0,
            use_leg_offset_for_distance=False,
        )
        # 8 directions per leg.
        assert len(cache) == 8
        # Each entry has the expected keys.
        for key, entry in cache.items():
            assert key[0] == 0  # leg_idx
            assert 'corridor' in entry
            assert 'shadow' in entry
            assert 'edge_geom' in entry

    def test_cancellation_short_circuits(self, mixin_with_progress):
        """A False return from _report_progress flips the cache's
        __cancelled__ flag and the function returns early."""
        line = LineString([(0, 0), (1000, 0)])
        from scipy.stats import norm
        # Make _report_progress return False on the 2nd call.
        calls = [True, False, False]

        def fake_progress(*args):
            return calls.pop(0) if calls else False

        mixin_with_progress._report_progress = fake_progress
        cache = mixin_with_progress._precompute_shadow_layer(
            transformed_lines=[line],
            distributions=[[norm(loc=0, scale=100.0)]],
            weights=[[1.0]],
            structures=[],
            depths=[],
            struct_min_dists=None,
            depth_min_dists=None,
            reach_distance=1_000.0,
            drift_repair={'use_lognormal': False, 'std': 1.0, 'loc': 0.0, 'scale': 1.0},
            drift_speed=1.0,
            use_leg_offset_for_distance=False,
        )
        # The single-leg path reports progress after every d_idx; once it
        # gets False it sets __cancelled__ and returns.
        assert cache.get('__cancelled__') is True

    def test_degenerate_leg_handled(self, mixin_with_progress):
        """A LineString with < 2 coords -> leg_state stays None -> shadow
        task uses the bounds-from-obstacles fallback."""
        line = LineString()  # empty
        struct = {'id': 's1', 'wkt': box(0, 0, 10, 10), 'height': 20.0}
        from scipy.stats import norm
        cache = mixin_with_progress._precompute_shadow_layer(
            transformed_lines=[line],
            distributions=[[]],
            weights=[[]],
            structures=[struct],
            depths=[],
            struct_min_dists=None,
            depth_min_dists=None,
            reach_distance=100.0,
            drift_repair={'use_lognormal': False, 'std': 1.0, 'loc': 0.0, 'scale': 1.0},
            drift_speed=1.0,
            use_leg_offset_for_distance=False,
        )
        # Doesn't crash -- returns a non-cancelled cache with 8 entries.
        assert '__cancelled__' not in cache
        assert len(cache) == 8

    def test_multi_leg_triggers_thread_pool(self, mixin_with_progress):
        """Multiple legs activate the ThreadPoolExecutor path and complete
        without cancellation."""
        from scipy.stats import norm
        lines = [
            LineString([(0, 0), (1_000, 0)]),
            LineString([(2_000, 0), (3_000, 0)]),
        ]
        cache = mixin_with_progress._precompute_shadow_layer(
            transformed_lines=lines,
            distributions=[[norm(loc=0, scale=100.0)]] * 2,
            weights=[[1.0]] * 2,
            structures=[],
            depths=[],
            struct_min_dists=None,
            depth_min_dists=None,
            reach_distance=500.0,
            drift_repair={'use_lognormal': False, 'std': 1.0, 'loc': 0.0, 'scale': 1.0},
            drift_speed=1.0,
            use_leg_offset_for_distance=False,
        )
        # 2 legs x 8 dirs = 16 entries.
        assert len(cache) == 16
        assert '__cancelled__' not in cache

    def test_unreachable_structure_skipped(self, mixin_with_progress):
        """struct_min_dists indicating an unreachable structure trips the
        _struct_reachable guard -> shadow not populated for that s_idx."""
        line = LineString([(0, 0), (1000, 0)])
        struct = {'id': 's_far', 'wkt': box(100_000, 100_000, 100_100, 100_100), 'height': 20.0}
        from scipy.stats import norm
        # struct_min_dists[leg_idx=0][math_dir=*][s_idx=0] = huge value
        # The code checks ``dist <= reach_distance * 1.01``; a huge dist trips this.
        struct_mins = [[[999_999.0] for _ in range(8)]]
        cache = mixin_with_progress._precompute_shadow_layer(
            transformed_lines=[line],
            distributions=[[norm(loc=0, scale=100.0)]],
            weights=[[1.0]],
            structures=[struct],
            depths=[],
            struct_min_dists=struct_mins,
            depth_min_dists=None,
            reach_distance=100.0,
            drift_repair={'use_lognormal': False, 'std': 1.0, 'loc': 0.0, 'scale': 1.0},
            drift_speed=1.0,
            use_leg_offset_for_distance=False,
        )
        # In each direction, the struct is not in the shadow dict.
        for (leg_idx, d_idx), entry in cache.items():
            if leg_idx == 0:
                assert ('allision', 0) not in entry['shadow']


# ---------------------------------------------------------------------------
# _precompute_bucket_memo direct invocation
# ---------------------------------------------------------------------------

class TestPrecomputeBucketMemo:
    def test_empty_traffic_returns_empty_memo(self, mixin_with_progress):
        """No traffic -> nothing to memo."""
        memo = mixin_with_progress._precompute_bucket_memo(
            data={'traffic_data': {}, 'drift': {'anchor_d': 0, 'anchor_p': 0}},
            transformed_lines=[],
            structures=[],
            depths=[],
            struct_min_dists=None,
            depth_min_dists=None,
            struct_probability_holes=[],
            depth_probability_holes=[],
            shadow_cache={},
            threshold_to_idx=None,
            reach_distance=0.0,
        )
        assert '__cancelled__' not in memo


# ---------------------------------------------------------------------------
# _precompute_spatial direct invocation
# ---------------------------------------------------------------------------

class TestPrecomputeSpatial:
    def test_empty_inputs_produces_empty_lists(self, mixin_with_progress):
        """No structs / depths -> all four returned lists are empty."""
        out = mixin_with_progress._precompute_spatial(
            transformed_lines=[],
            distributions=[],
            weights=[],
            structs_gdfs=[],
            depths_gdfs=[],
            reach_distance=0.0,
            data={'use_analytical': True},
        )
        struct_min, depth_min, struct_prob, depth_prob = out
        assert struct_min == []
        assert depth_min == []
        assert struct_prob == []
        assert depth_prob == []

    def test_monte_carlo_branch_selected_when_use_analytical_false(
        self, mixin_with_progress, monkeypatch
    ):
        """data['use_analytical']=False routes through the Monte Carlo code
        path (compute_probability_holes)."""
        import compute.drifting_model as mod
        called = []
        monkeypatch.setattr(
            mod, 'compute_probability_holes',
            lambda *a, **k: called.append('mc') or [],
        )
        monkeypatch.setattr(
            mod, 'compute_probability_holes_analytical',
            lambda *a, **k: called.append('analytical') or [],
        )
        # Provide one tiny depths_gdf so the compute_holes_fn is invoked.
        import geopandas as gpd
        from shapely.geometry import Polygon
        depths_gdf = gpd.GeoDataFrame(
            {'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]})
        mixin_with_progress._precompute_spatial(
            transformed_lines=[LineString([(0, 0), (10, 0)])],
            distributions=[[]],
            weights=[[]],
            structs_gdfs=[],
            depths_gdfs=[depths_gdf],
            reach_distance=100.0,
            data={'use_analytical': False},
        )
        assert 'mc' in called
        assert 'analytical' not in called


# ---------------------------------------------------------------------------
# run_drifting_model -- cancellation and early-return paths
# ---------------------------------------------------------------------------

class TestRunDriftingModelCancellationPaths:
    def _mock_parent(self):
        mp = MagicMock()
        mp.main_widget = MagicMock()
        mp.main_widget.LEPDriftAllision.setText = MagicMock()
        mp.main_widget.LEPDriftingGrounding.setText = MagicMock()
        mp.main_widget.cbShipCategories.count = MagicMock(return_value=0)
        mp.main_widget.LEReportPath.text = MagicMock(return_value='')
        return mp

    def test_shadow_cancellation_returns_zeros(self, monkeypatch):
        """When _precompute_shadow_layer sets __cancelled__, run_drifting_model
        short-circuits and returns (0.0, 0.0)."""
        from compute.run_calculations import Calculation
        from compute.basic_equations import default_blackout_by_ship_type

        # Import the minimal project data builder from test_cascade_minimal.
        from tests.test_cascade_minimal import _build_minimal_project

        data = copy.deepcopy(_build_minimal_project())
        data['drift'].setdefault(
            'blackout_by_ship_type', default_blackout_by_ship_type())

        calc = Calculation(self._mock_parent())
        calc.set_progress_callback(lambda c, t, m: True)

        # Patch _precompute_shadow_layer on the Calculation instance to return
        # an immediately-cancelled cache.
        def fake_shadow(*args, **kwargs):
            return {'__cancelled__': True}

        monkeypatch.setattr(calc, '_precompute_shadow_layer', fake_shadow)

        a, g = calc.run_drifting_model(data)
        assert a == 0.0 and g == 0.0

    def test_bucket_memo_cancellation_returns_zeros(self, monkeypatch):
        """When _precompute_bucket_memo signals cancellation, same short-circuit."""
        from compute.run_calculations import Calculation
        from compute.basic_equations import default_blackout_by_ship_type
        from tests.test_cascade_minimal import _build_minimal_project

        data = copy.deepcopy(_build_minimal_project())
        data['drift'].setdefault(
            'blackout_by_ship_type', default_blackout_by_ship_type())

        calc = Calculation(self._mock_parent())
        calc.set_progress_callback(lambda c, t, m: True)

        monkeypatch.setattr(
            calc, '_precompute_bucket_memo',
            lambda *a, **k: {'__cancelled__': True},
        )
        a, g = calc.run_drifting_model(data)
        assert a == 0.0 and g == 0.0

    def test_debug_trace_populates_report(self, monkeypatch):
        """Setting drift.debug_trace=True populates report['debug_obstacles']
        with per-obstacle aggregated metrics."""
        from compute.run_calculations import Calculation
        from compute.basic_equations import default_blackout_by_ship_type
        from tests.test_cascade_minimal import _build_minimal_project

        data = copy.deepcopy(_build_minimal_project())
        data['drift'].setdefault(
            'blackout_by_ship_type', default_blackout_by_ship_type())
        data['drift']['debug_trace'] = True

        calc = Calculation(self._mock_parent())
        calc.set_progress_callback(lambda c, t, m: True)
        calc.run_drifting_model(data)

        dbg = calc.drifting_report.get('debug_obstacles', {})
        assert isinstance(dbg, dict)
        # Expect at least some debug entries from the cascade.
        assert len(dbg) > 0
        for rec in dbg.values():
            assert 'leg_dir_key' in rec
            assert 'obstacle' in rec
            assert 'type' in rec
            assert rec['count'] > 0

    def test_multipolygon_depth_splits(self, monkeypatch):
        """A depth stored as a MultiPolygon WKT is split into per-polygon
        entries inside _build_transformed (L953-963)."""
        from compute.run_calculations import Calculation
        from compute.basic_equations import default_blackout_by_ship_type
        from tests.test_cascade_minimal import _build_minimal_project

        data = copy.deepcopy(_build_minimal_project())
        data['drift'].setdefault(
            'blackout_by_ship_type', default_blackout_by_ship_type())
        # Replace the single depth with a MultiPolygon.
        data['depths'] = [[
            'multi_d', '12',
            'MULTIPOLYGON(((14.08 55.22, 14.10 55.22, 14.10 55.23, '
            '14.08 55.23, 14.08 55.22)),'
            '((14.12 55.21, 14.14 55.21, 14.14 55.22, '
            '14.12 55.22, 14.12 55.21)))',
        ]]
        # Same for objects.
        data['objects'] = [[
            'multi_s', '20',
            'MULTIPOLYGON(((14.09 55.208, 14.10 55.208, 14.10 55.212, '
            '14.09 55.212, 14.09 55.208)),'
            '((14.11 55.210, 14.115 55.210, 14.115 55.212, '
            '14.11 55.212, 14.11 55.210)))',
        ]]

        calc = Calculation(self._mock_parent())
        calc.set_progress_callback(lambda c, t, m: True)
        a, g = calc.run_drifting_model(data)
        assert a >= 0.0 and g >= 0.0

    def test_merged_depths_path(self, monkeypatch):
        """Multiple depth polygons with distinct values trigger the merged
        polygon + threshold_to_idx path."""
        from compute.run_calculations import Calculation
        from compute.basic_equations import default_blackout_by_ship_type
        from tests.test_cascade_minimal import _build_minimal_project

        data = copy.deepcopy(_build_minimal_project())
        data['drift'].setdefault(
            'blackout_by_ship_type', default_blackout_by_ship_type())
        # Add multiple depths with different values so merging kicks in.
        # use_merged = len(depths) > len(unique_depth_vals) + 1 AND
        #              len(unique_depth_vals) > 0.
        # With 4 depths at 2 unique values -> 4 > 3 -> use_merged=True.
        poly_tpl = (
            'POLYGON((14.{a} 55.22, 14.{b} 55.22, 14.{b} 55.23, '
            '14.{a} 55.23, 14.{a} 55.22))'
        )
        data['depths'] = [
            ['d1', '10', poly_tpl.format(a='08', b='09')],
            ['d2', '10', poly_tpl.format(a='11', b='12')],
            ['d3', '12', poly_tpl.format(a='13', b='14')],
            ['d4', '12', poly_tpl.format(a='15', b='16')],
        ]

        calc = Calculation(self._mock_parent())
        calc.set_progress_callback(lambda c, t, m: True)

        # Ensure it runs without raising.
        a, g = calc.run_drifting_model(data)
        assert a >= 0.0 and g >= 0.0

    def test_empty_traffic_returns_zero(self):
        """Missing traffic_data short-circuits to 0.0 / 0.0."""
        from compute.run_calculations import Calculation
        calc = Calculation(self._mock_parent())
        a, g = calc.run_drifting_model({})
        assert a == 0.0 and g == 0.0

    def test_grounding_widget_setter_swallows_exception(self, monkeypatch):
        """If the grounding-widget setText raises, the early-return path
        keeps going and still produces 0.0 / 0.0."""
        from compute.run_calculations import Calculation
        parent = self._mock_parent()
        parent.main_widget.LEPDriftingGrounding.setText.side_effect = RuntimeError('x')
        calc = Calculation(parent)
        a, g = calc.run_drifting_model({})
        assert a == 0.0 and g == 0.0

    def test_empty_structs_and_depths_returns_zero(self, monkeypatch):
        """No structs and no depths -> short-circuit at L1628."""
        from compute.run_calculations import Calculation
        from compute.basic_equations import default_blackout_by_ship_type
        from tests.test_cascade_minimal import _build_minimal_project

        data = copy.deepcopy(_build_minimal_project())
        data['drift'].setdefault(
            'blackout_by_ship_type', default_blackout_by_ship_type())
        data['depths'] = []
        data['objects'] = []

        calc = Calculation(self._mock_parent())
        calc.set_progress_callback(lambda c, t, m: True)
        a, g = calc.run_drifting_model(data)
        assert a == 0.0 and g == 0.0

    def test_create_result_layers_exception_swallowed(self, monkeypatch):
        """Layer-creation failure doesn't propagate; calc returns values as usual."""
        from compute.run_calculations import Calculation
        from compute.basic_equations import default_blackout_by_ship_type
        from tests.test_cascade_minimal import _build_minimal_project
        import compute.drifting_model as mod

        data = copy.deepcopy(_build_minimal_project())
        data['drift'].setdefault(
            'blackout_by_ship_type', default_blackout_by_ship_type())

        calc = Calculation(self._mock_parent())
        calc.set_progress_callback(lambda c, t, m: True)

        def boom(*a, **k):
            raise RuntimeError('synthetic layer failure')

        monkeypatch.setattr(mod, 'create_result_layers', boom)

        a, g = calc.run_drifting_model(data)
        # Should have completed successfully despite layer error.
        assert a >= 0.0 and g >= 0.0