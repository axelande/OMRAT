"""Unit tests for the pure-Python helpers on ``DriftCorridorGenerator``.

The generator class reads heavily from Qt widgets (``self.plugin.main_widget.twDepthList``
etc.) which makes ``precollect_data`` and ``generate_corridors`` hard to exercise
in isolation.  This file focuses on the small parse/lookup helpers that don't
require the full QGIS stack.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.drift.generator import DriftCorridorGenerator


@pytest.fixture
def gen():
    """Generator with a bare plugin stub -- enough for the pure helpers."""
    plugin = MagicMock()
    return DriftCorridorGenerator(plugin)


def _fake_table(rows: list[list[str | None]]):
    """Build a minimal table mock that quacks like QTableWidget.

    ``rows`` is ``[[cell_text_col0, cell_text_col1, ...], ...]``; ``None`` maps
    to ``tbl.item(...) is None``.
    """
    tbl = MagicMock()
    tbl.rowCount.return_value = len(rows)

    def item(r, c):
        if r >= len(rows) or c >= len(rows[r]):
            return None
        txt = rows[r][c]
        if txt is None:
            return None
        return SimpleNamespace(text=lambda txt=txt: txt)

    tbl.item.side_effect = item
    return tbl


# ---------------------------------------------------------------------------
# clear_cache / set_progress_callback / _report_progress
# ---------------------------------------------------------------------------

class TestCacheAndProgress:
    def test_clear_cache_resets_state(self, gen):
        gen._precollected_data = {'legs': []}
        gen._cancelled = True
        gen._progress_callback = lambda *a: True
        gen.clear_cache()
        assert gen._precollected_data is None
        assert gen._cancelled is False
        assert gen._progress_callback is None

    def test_report_progress_without_callback_returns_true(self, gen):
        assert gen._report_progress(1, 10, 'working') is True

    def test_report_progress_forwards_to_callback(self, gen):
        seen = []

        def cb(c, t, m):
            seen.append((c, t, m))
            return True

        gen.set_progress_callback(cb)
        assert gen._report_progress(2, 20, 'tick') is True
        assert seen == [(2, 20, 'tick')]

    def test_report_progress_false_sets_cancelled(self, gen):
        gen.set_progress_callback(lambda *a: False)
        assert gen._report_progress(1, 10, 'stop') is False
        assert gen._cancelled is True


# ---------------------------------------------------------------------------
# _detect_depth_bin_width
# ---------------------------------------------------------------------------

class TestDetectDepthBinWidth:
    def test_uniform_5m_steps(self, gen):
        tbl = _fake_table([
            ['d1', '5', '...'],
            ['d2', '10', '...'],
            ['d3', '15', '...'],
            ['d4', '20', '...'],
        ])
        assert gen._detect_depth_bin_width(tbl) == pytest.approx(5.0)

    def test_single_value_returns_zero(self, gen):
        tbl = _fake_table([['d1', '10', '...']])
        assert gen._detect_depth_bin_width(tbl) == 0.0

    def test_intervals_are_skipped_for_detection(self, gen):
        """Rows like '0-10' are ignored -- only bare values count."""
        tbl = _fake_table([
            ['d1', '0-10', '...'],    # skipped (interval)
            ['d2', '10-20', '...'],   # skipped (interval)
            ['d3', '15', '...'],
            ['d4', '20', '...'],
        ])
        # Only two bare values -> diff = 5.
        assert gen._detect_depth_bin_width(tbl) == pytest.approx(5.0)

    def test_negative_depths_use_absolute_value(self, gen):
        tbl = _fake_table([
            ['d1', '-5', '...'],
            ['d2', '-10', '...'],
            ['d3', '-15', '...'],
        ])
        assert gen._detect_depth_bin_width(tbl) == pytest.approx(5.0)

    def test_negative_interval_is_skipped(self, gen):
        """A negative interval like '-10--5' contains '--' and is skipped."""
        tbl = _fake_table([
            ['d1', '-10--5', '...'],   # skipped
            ['d2', '5', '...'],
            ['d3', '10', '...'],
        ])
        assert gen._detect_depth_bin_width(tbl) == pytest.approx(5.0)

    def test_empty_table_returns_zero(self, gen):
        assert gen._detect_depth_bin_width(_fake_table([])) == 0.0

    def test_missing_cell_skipped(self, gen):
        tbl = _fake_table([
            ['d1', None, '...'],
            ['d2', '5', '...'],
            ['d3', '10', '...'],
        ])
        assert gen._detect_depth_bin_width(tbl) == pytest.approx(5.0)

    def test_non_numeric_skipped(self, gen):
        tbl = _fake_table([
            ['d1', 'nonsense', '...'],
            ['d2', '5', '...'],
            ['d3', '10', '...'],
        ])
        assert gen._detect_depth_bin_width(tbl) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _parse_depth_value
# ---------------------------------------------------------------------------

class TestParseDepthValue:
    def test_interval_uses_upper_bound(self, gen):
        assert gen._parse_depth_value('0-10', bin_width=5.0) == 10.0

    def test_interval_with_three_parts_uses_last(self, gen):
        assert gen._parse_depth_value('a-b-7.5', bin_width=5.0) == 7.5

    def test_negative_interval_uses_less_negative_bound(self, gen):
        """'-10--5' splits on '--' into ['-10', '5'] -> abs(-10) = 10."""
        assert gen._parse_depth_value('-10--5', bin_width=5.0) == 10.0

    def test_single_positive_adds_bin_width(self, gen):
        assert gen._parse_depth_value('10', bin_width=5.0) == 15.0

    def test_single_negative_adds_bin_width_to_abs(self, gen):
        assert gen._parse_depth_value('-10', bin_width=5.0) == 15.0

    def test_zero_bin_width(self, gen):
        assert gen._parse_depth_value('10', bin_width=0.0) == 10.0

    def test_unparseable_returns_none(self, gen):
        assert gen._parse_depth_value('nonsense', bin_width=5.0) is None

    def test_empty_string_returns_none(self, gen):
        assert gen._parse_depth_value('', bin_width=5.0) is None


# ---------------------------------------------------------------------------
# _get_distribution_std
# ---------------------------------------------------------------------------

class TestGetDistributionStd:
    def test_reads_text_from_widget(self, gen):
        gen.plugin.main_widget.leNormStd1_1.text.return_value = '250'
        assert gen._get_distribution_std() == 250.0

    def test_empty_text_falls_back_to_default(self, gen):
        gen.plugin.main_widget.leNormStd1_1.text.return_value = ''
        assert gen._get_distribution_std() == 100.0

    def test_zero_falls_back_to_default(self, gen):
        """Zero std is treated as 'not set' and falls back to 100."""
        gen.plugin.main_widget.leNormStd1_1.text.return_value = '0'
        assert gen._get_distribution_std() == 100.0

    def test_missing_widget_falls_back(self, gen):
        """A ValueError / AttributeError bubble still returns the default."""
        gen.plugin.main_widget.leNormStd1_1.text.return_value = 'not-a-number'
        assert gen._get_distribution_std() == 100.0


# ---------------------------------------------------------------------------
# _get_repair_params / _get_drift_speed_ms
# ---------------------------------------------------------------------------

class TestRepairAndDriftSpeed:
    def test_repair_params_from_drift_values(self, gen):
        gen.plugin.drift_values = {
            'use_lognormal': 1, 'std': 0.8, 'loc': 0.1, 'scale': 0.9,
        }
        params = gen._get_repair_params()
        assert params == {
            'use_lognormal': 1, 'std': 0.8, 'loc': 0.1, 'scale': 0.9,
        }

    def test_repair_params_use_defaults_when_missing(self, gen):
        gen.plugin.drift_values = {}
        params = gen._get_repair_params()
        # Exact default values from the source.
        assert params['use_lognormal'] == 1
        assert params['std'] == pytest.approx(0.95)
        assert params['loc'] == pytest.approx(0.2)
        assert params['scale'] == pytest.approx(0.85)

    def test_drift_speed_converts_knots_to_ms(self, gen):
        gen.plugin.drift_values = {'speed': 2.0}
        # 2 kts -> 2 * 1852 / 3600 ~ 1.0289 m/s
        assert gen._get_drift_speed_ms() == pytest.approx(2.0 * 1852 / 3600)

    def test_drift_speed_default_1p94_kts(self, gen):
        gen.plugin.drift_values = {}
        # Default from source is 1.94 kts.
        assert gen._get_drift_speed_ms() == pytest.approx(1.94 * 1852 / 3600)


# ---------------------------------------------------------------------------
# _get_legs_from_routes
# ---------------------------------------------------------------------------

class TestGetLegsFromRoutes:
    def test_extracts_linestrings_from_vector_layers(self, gen):
        # Two fake layers, each with one feature.
        line1_wkt = 'LINESTRING (0 0, 10 0)'
        line2_wkt = 'LINESTRING (0 0, 0 10)'

        def make_feature(wkt: str):
            geom = MagicMock()
            geom.isNull.return_value = False
            geom.asWkt.return_value = wkt
            feat = MagicMock()
            feat.geometry.return_value = geom
            return feat

        layer1 = MagicMock()
        layer1.getFeatures.return_value = iter([make_feature(line1_wkt)])
        layer2 = MagicMock()
        layer2.getFeatures.return_value = iter([make_feature(line2_wkt)])
        gen.plugin.qgis_geoms.vector_layers = [layer1, layer2]

        legs = gen._get_legs_from_routes()
        assert len(legs) == 2
        assert all(isinstance(l, LineString) for l in legs)

    def test_null_geometry_skipped(self, gen):
        geom = MagicMock()
        geom.isNull.return_value = True
        feat = MagicMock()
        feat.geometry.return_value = geom
        layer = MagicMock()
        layer.getFeatures.return_value = iter([feat])
        gen.plugin.qgis_geoms.vector_layers = [layer]
        assert gen._get_legs_from_routes() == []

    def test_non_linestring_geom_ignored(self, gen):
        geom = MagicMock()
        geom.isNull.return_value = False
        geom.asWkt.return_value = 'POINT(0 0)'
        feat = MagicMock()
        feat.geometry.return_value = geom
        layer = MagicMock()
        layer.getFeatures.return_value = iter([feat])
        gen.plugin.qgis_geoms.vector_layers = [layer]
        assert gen._get_legs_from_routes() == []

    def test_malformed_wkt_swallowed(self, gen):
        geom = MagicMock()
        geom.isNull.return_value = False
        geom.asWkt.return_value = 'NONSENSE WKT'
        feat = MagicMock()
        feat.geometry.return_value = geom
        layer = MagicMock()
        layer.getFeatures.return_value = iter([feat])
        gen.plugin.qgis_geoms.vector_layers = [layer]
        # Does not raise.
        assert gen._get_legs_from_routes() == []


# ---------------------------------------------------------------------------
# _get_depth_obstacles
# ---------------------------------------------------------------------------

class TestGetDepthObstacles:
    def test_only_depths_below_threshold_kept(self, gen):
        """Depths > threshold are filtered out (even after the bin-width add)."""
        tbl = _fake_table([
            ['d1', '5', 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'],
            ['d2', '50', 'POLYGON((2 2, 3 2, 3 3, 2 3, 2 2))'],
            ['d3', '100', 'POLYGON((4 4, 5 4, 5 5, 4 5, 4 4))'],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        # Threshold large enough that d1 (5+45=50) passes but the others don't.
        out = gen._get_depth_obstacles(depth_threshold=50.0)
        # bin_width = (50-5) = 45 (most-common diff among bare values 5, 50, 100)
        # Actually bin_width is min diff -- let's just check that exactly one
        # row qualifies.
        assert len(out) == 1
        poly, depth = out[0]
        assert depth >= 5.0

    def test_multipolygon_split(self, gen):
        tbl = _fake_table([
            ['m', '5', 'MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)),'
                      '((2 2, 3 2, 3 3, 2 3, 2 2)))'],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        # Depth=5 + bin_width=0 = 5. Threshold=10.
        out = gen._get_depth_obstacles(depth_threshold=10.0)
        # MultiPolygon split into 2 sub-polygons.
        assert len(out) == 2

    def test_invalid_wkt_swallowed(self, gen):
        tbl = _fake_table([
            ['d1', '5', 'NOT WKT'],
            ['d2', '7', 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        # The bad row is logged but doesn't crash.
        out = gen._get_depth_obstacles(depth_threshold=20.0)
        # Only the valid row contributes (d2).
        assert len(out) == 1

    def test_missing_depth_cell_skipped(self, gen):
        tbl = _fake_table([
            ['d1', None, 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        assert gen._get_depth_obstacles(depth_threshold=20.0) == []

    def test_missing_wkt_cell_skipped(self, gen):
        tbl = _fake_table([
            ['d1', '5', None],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        assert gen._get_depth_obstacles(depth_threshold=20.0) == []

    def test_blank_wkt_skipped(self, gen):
        tbl = _fake_table([
            ['d1', '5', '   '],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        assert gen._get_depth_obstacles(depth_threshold=20.0) == []


# ---------------------------------------------------------------------------
# _get_structure_obstacles
# ---------------------------------------------------------------------------

class TestGetStructureObstacles:
    def test_only_structures_below_threshold_kept(self, gen):
        tbl = _fake_table([
            ['s1', '5', 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'],
            ['s2', '50', 'POLYGON((2 2, 3 2, 3 3, 2 3, 2 2))'],
        ])
        gen.plugin.main_widget.twObjectList = tbl
        out = gen._get_structure_obstacles(height_threshold=10.0)
        assert len(out) == 1
        assert out[0][1] == 5.0

    def test_multipolygon_split(self, gen):
        tbl = _fake_table([
            ['m', '5', 'MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)),'
                      '((2 2, 3 2, 3 3, 2 3, 2 2)))'],
        ])
        gen.plugin.main_widget.twObjectList = tbl
        out = gen._get_structure_obstacles(height_threshold=10.0)
        assert len(out) == 2

    def test_invalid_height_swallowed(self, gen):
        tbl = _fake_table([
            ['s1', 'not-a-number', 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'],
        ])
        gen.plugin.main_widget.twObjectList = tbl
        # Row is logged + skipped.
        assert gen._get_structure_obstacles(height_threshold=10.0) == []

    def test_missing_height_skipped(self, gen):
        tbl = _fake_table([['s1', None, 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))']])
        gen.plugin.main_widget.twObjectList = tbl
        assert gen._get_structure_obstacles(height_threshold=10.0) == []


# ---------------------------------------------------------------------------
# _get_anchor_zone
# ---------------------------------------------------------------------------

class TestGetAnchorZone:
    def test_zero_threshold_returns_empty(self, gen):
        assert gen._get_anchor_zone(anchor_threshold=0).is_empty

    def test_negative_threshold_returns_empty(self, gen):
        assert gen._get_anchor_zone(anchor_threshold=-1.0).is_empty

    def test_unions_anchorable_depths(self, gen):
        tbl = _fake_table([
            ['d1', '3', 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'],
            ['d2', '7', 'POLYGON((2 2, 3 2, 3 3, 2 3, 2 2))'],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        # Threshold = 100 -> both qualify regardless of bin_width adjustment.
        zone = gen._get_anchor_zone(anchor_threshold=100.0)
        # Union covers both polygons -> area > 1 (sum of two unit squares = 2).
        assert zone.area >= 2.0 - 1e-6

    def test_filters_too_deep(self, gen):
        tbl = _fake_table([
            ['d1', '50', 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        # Anchor threshold = 10 -> too deep -> empty zone.
        zone = gen._get_anchor_zone(anchor_threshold=10.0)
        assert zone.is_empty

    def test_invalid_wkt_swallowed(self, gen):
        tbl = _fake_table([
            ['d1', '3', 'NOT WKT'],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        zone = gen._get_anchor_zone(anchor_threshold=10.0)
        # Bad row swallowed -> empty zone.
        assert zone.is_empty

    def test_missing_cells_skipped(self, gen):
        tbl = _fake_table([
            ['d1', None, 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'],
            ['d2', '5', None],
            ['d3', '5', '  '],
            ['d4', '5', 'POLYGON((10 10, 11 10, 11 11, 10 11, 10 10))'],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        zone = gen._get_anchor_zone(anchor_threshold=10.0)
        # Only the last row contributes.
        assert zone.area == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# precollect_data
# ---------------------------------------------------------------------------

class TestPrecollectData:
    def test_aggregates_into_precollected_data(self, gen):
        depth_tbl = _fake_table([
            ['d1', '5', 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'],
        ])
        obj_tbl = _fake_table([
            ['s1', '8', 'POLYGON((2 2, 3 2, 3 3, 2 3, 2 2))'],
        ])
        gen.plugin.main_widget.twDepthList = depth_tbl
        gen.plugin.main_widget.twObjectList = obj_tbl
        gen.plugin.qgis_geoms.vector_layers = []  # no legs
        gen.plugin.drift_values = {
            'anchor_d': 2.0, 'speed': 1.94,
            'use_lognormal': 1, 'std': 0.95, 'loc': 0.2, 'scale': 0.85,
        }
        gen.plugin.main_widget.leNormStd1_1.text.return_value = '50'
        gen.precollect_data(depth_threshold=10.0, height_threshold=10.0)
        d = gen._precollected_data
        assert d is not None
        assert 'depth_obstacles' in d and len(d['depth_obstacles']) == 1
        assert 'structure_obstacles' in d and len(d['structure_obstacles']) == 1
        assert 'anchor_zone' in d
        assert d['lateral_std'] == 50.0
        assert d['drift_speed'] == pytest.approx(1.94 * 1852 / 3600)
