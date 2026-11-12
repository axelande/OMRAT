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

from geometries.drift.generator import DriftCorridorGenerator  # noqa: E402


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
# _parse_depth_value: the model's class value (depth_simplify.depth_value)
# ---------------------------------------------------------------------------

class TestParseDepthValue:
    """Until 2026-11-12 the corridors used an interval's deeper end, added
    the table's step to a single value and made negative depths positive."""

    def test_interval_uses_the_shallower_end(self, gen):
        assert gen._parse_depth_value('0-10') == 0.0
        assert gen._parse_depth_value('48-51') == 48.0

    def test_negative_interval_uses_the_shallower_end(self, gen):
        assert gen._parse_depth_value('-10--5') == -10.0

    def test_single_values_are_taken_as_they_are(self, gen):
        assert gen._parse_depth_value('10') == 10.0
        assert gen._parse_depth_value('-10') == -10.0
        assert gen._parse_depth_value('7.5') == 7.5

    def test_unparseable_returns_none(self, gen):
        assert gen._parse_depth_value('nonsense') is None
        assert gen._parse_depth_value('a-b-7.5') is None

    def test_empty_string_returns_none(self, gen):
        assert gen._parse_depth_value('') is None


# ---------------------------------------------------------------------------
# Corridor width and length: the drift model's
# ---------------------------------------------------------------------------

def _seg(sp, ep, std1=200.0, std2=200.0):
    return {'Start_Point': sp, 'End_Point': ep,
            'std1_1': std1, 'weight1_1': 100.0, 'std2_1': std2, 'weight2_1': 100.0}


class TestModelWidthAndLength:
    """Until 2026-11-12 every leg got the width of the leg shown on the
    Distributions tab and the length came from repair keys read at the wrong
    level of ``drift_values`` (always the defaults)."""

    def test_legs_and_half_widths_follow_segment_data(self, gen):
        from geometries.depth_simplify import lateral_spread_m
        gen.plugin.segment_data = {
            '1': _seg('11.0 56.0', '11.1 56.0', std1=200.0, std2=300.0),
            '2': _seg('11.1 56.0', '11.2 56.1', std1=50.0, std2=50.0),
        }
        legs, widths = gen._get_legs_and_half_widths()
        assert [list(leg.coords) for leg in legs] == [[(11.0, 56.0), (11.1, 56.0)], [(11.1, 56.0), (11.2, 56.1)]]
        seg1 = gen.plugin.segment_data['1']
        assert widths[0] == pytest.approx(max(lateral_spread_m(seg1, 0), lateral_spread_m(seg1, 1)))
        assert widths == pytest.approx([5 * 300.0, 5 * 50.0])

    def test_without_segment_data_uses_route_layers_and_fallback(self, gen, monkeypatch):
        from shapely.geometry import LineString
        from geometries.depth_simplify import FALLBACK_SPREAD_M
        gen.plugin.segment_data = {}
        monkeypatch.setattr(gen, '_get_legs_from_routes', lambda: [LineString([(0, 0), (1, 0)])])
        legs, widths = gen._get_legs_and_half_widths()
        assert len(legs) == 1 and widths == [FALLBACK_SPREAD_M]

    def test_length_is_the_model_reach(self, gen):
        from compute.drift_reach import REACH_QUANTILE, drift_reach_m
        gen.plugin.segment_data = {'1': _seg('11.0 56.0', '11.5 56.0')}
        gen.plugin.drift_values = {'speed': 1.0, 'repair': {
            'use_lognormal': True, 'std': 0.95, 'loc': 0.2, 'scale': 0.85, 'func': ''}}
        assert gen._get_corridor_length() == pytest.approx(drift_reach_m(gen.plugin.drift_values, REACH_QUANTILE))

    def test_length_capped_at_ten_legs(self, gen):
        from geometries.route_validation import leg_length_m
        seg = _seg('11.0 56.0', '11.001 56.0')
        gen.plugin.segment_data = {'1': seg}
        gen.plugin.drift_values = {'speed': 1.0, 'repair': {
            'use_lognormal': True, 'std': 0.95, 'loc': 0.2, 'scale': 0.85, 'func': ''}}
        assert gen._get_corridor_length() == pytest.approx(10 * leg_length_m(seg))


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
        assert all(isinstance(leg, LineString) for leg in legs)

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
        """Depths > threshold are filtered out; a class is its shallower end."""
        tbl = _fake_table([
            ['d1', '5', 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'],
            ['d2', '50-60', 'POLYGON((2 2, 3 2, 3 3, 2 3, 2 2))'],
            ['d3', '100', 'POLYGON((4 4, 5 4, 5 5, 4 5, 4 4))'],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        out = gen._get_depth_obstacles(depth_threshold=50.0)
        assert sorted(depth for _poly, depth in out) == [5.0, 50.0]

    def test_multipolygon_split(self, gen):
        tbl = _fake_table([
            ['m', '5', 'MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)),'
                       '((2 2, 3 2, 3 3, 2 3, 2 2)))'],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        # Depth 5 <= threshold 10.
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
        # Threshold = 100 -> both qualify.
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
        gen.plugin.qgis_geoms.vector_layers = []
        gen.plugin.segment_data = {'1': _seg('11.0 56.0', '11.1 56.0', std1=50.0, std2=50.0)}
        gen.plugin.drift_values = {
            'anchor_d': 2.0, 'speed': 1.94,
            'repair': {'use_lognormal': True, 'std': 0.95, 'loc': 0.2, 'scale': 0.85, 'func': ''},
        }
        gen.precollect_data(depth_threshold=10.0, height_threshold=10.0)
        d = gen._precollected_data
        assert d is not None
        assert 'depth_obstacles' in d and len(d['depth_obstacles']) == 1
        assert 'structure_obstacles' in d and len(d['structure_obstacles']) == 1
        assert 'anchor_zone' in d
        assert len(d['legs']) == 1 and d['half_widths'] == [pytest.approx(250.0)]
        assert d['projection_dist'] == pytest.approx(gen._get_corridor_length())


# ---------------------------------------------------------------------------
# UserRole WKT (tables show "Polygon(N pts)" and keep the WKT in UserRole)
# ---------------------------------------------------------------------------

def _fake_table_with_roles(rows: list[list[str | tuple[str, str] | None]]):
    """Like ``_fake_table`` but a ``(display, user_role)`` tuple builds a cell
    whose ``text()`` is the summary and ``data(256)`` the full payload --
    mirroring ``omrat_utils.handle_object._wkt_table_item``."""
    tbl = MagicMock()
    tbl.rowCount.return_value = len(rows)

    def item(r, c):
        if r >= len(rows) or c >= len(rows[r]):
            return None
        cell = rows[r][c]
        if cell is None:
            return None
        if isinstance(cell, tuple):
            display, payload = cell
            return SimpleNamespace(text=lambda d=display: d,
                                   data=lambda role, p=payload: p if role == 256 else None)
        return SimpleNamespace(text=lambda t=cell: t)

    tbl.item.side_effect = item
    return tbl


class TestUserRoleWkt:
    SQUARE = 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'

    def test_depth_obstacle_read_from_user_role(self, gen):
        """Regression: the summary text is not WKT; the real geometry lives in
        UserRole.  Before the fix every row failed to parse and the corridors
        were never clipped at shallow water."""
        tbl = _fake_table_with_roles([
            ['d1', '5', ('Polygon(5 pts)', self.SQUARE)],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        out = gen._get_depth_obstacles(depth_threshold=10.0)
        assert len(out) == 1
        poly, depth = out[0]
        assert poly.area == pytest.approx(1.0)
        assert depth == 5.0

    def test_structure_obstacle_read_from_user_role(self, gen):
        tbl = _fake_table_with_roles([
            ['s1', '5', ('Polygon(5 pts)', self.SQUARE)],
        ])
        gen.plugin.main_widget.twObjectList = tbl
        out = gen._get_structure_obstacles(height_threshold=10.0)
        assert len(out) == 1
        assert out[0][0].area == pytest.approx(1.0)

    def test_anchor_zone_read_from_user_role(self, gen):
        tbl = _fake_table_with_roles([
            ['d1', '5', ('Polygon(5 pts)', self.SQUARE)],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        zone = gen._get_anchor_zone(anchor_threshold=10.0)
        assert zone.area == pytest.approx(1.0)

    def test_plain_text_cell_still_works(self, gen):
        """Cells without a UserRole payload (short WKT, legacy items) keep
        reading the display text."""
        tbl = _fake_table_with_roles([
            ['d1', '5', self.SQUARE],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        assert len(gen._get_depth_obstacles(depth_threshold=10.0)) == 1

    def test_empty_user_role_falls_back_to_text(self, gen):
        tbl = _fake_table_with_roles([
            ['d1', '5', (self.SQUARE, '')],
        ])
        gen.plugin.main_widget.twDepthList = tbl
        assert len(gen._get_depth_obstacles(depth_threshold=10.0)) == 1
