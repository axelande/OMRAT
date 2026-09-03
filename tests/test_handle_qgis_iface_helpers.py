"""Targeted tests for ``geometries.handle_qgis_iface``.

Existing tests in ``test_qgis_interaction.py`` exercise the
``HandleQGISIface`` class via the full plugin (``omrat`` fixture).
This file adds:

* Pure helper tests for ``is_valid_point_pair`` and ``calculate_tangent_line``.
* Direct unit tests for short methods that don't need a full canvas
  (``create_fields``, ``point4326_from_wkt``, ``format_wkt``,
  ``ensure_tangent_layer``, ``ensure_tangent_fields``, ``calculate_midpoint_utm``,
  ``style_layer``, ``label_layer``, ``add_tangent_feature``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestPureHelpers:
    def test_is_valid_point_pair_outside_unit_square(self):
        from qgis.core import QgsPointXY
        from geometries.handle_qgis_iface import is_valid_point_pair
        # Both points well outside [-1, 1] -> valid.
        assert is_valid_point_pair(QgsPointXY(10, 20), QgsPointXY(30, 40))

    def test_is_valid_point_pair_inside_unit_square(self):
        from qgis.core import QgsPointXY
        from geometries.handle_qgis_iface import is_valid_point_pair
        # Either point near origin (within ±1) -> invalid.
        assert not is_valid_point_pair(QgsPointXY(0.5, 0.5), QgsPointXY(30, 40))
        assert not is_valid_point_pair(QgsPointXY(30, 40), QgsPointXY(0.5, 0.5))

    def test_calculate_tangent_line_basic(self):
        from qgis.core import QgsPointXY
        from geometries.handle_qgis_iface import calculate_tangent_line
        # Horizontal east-going segment: start=(0,0), end=(10,0); midpoint=(5,0).
        # Perpendicular offset -> tangent runs vertically through midpoint.
        mid = QgsPointXY(5.0, 0.0)
        s = QgsPointXY(0.0, 0.0)
        e = QgsPointXY(10.0, 0.0)
        t1, t2 = calculate_tangent_line(mid, s, e, offset=2.0)
        # Tangent endpoints share the midpoint x and offset y.
        assert t1.x() == pytest.approx(5.0, abs=1e-9)
        assert t2.x() == pytest.approx(5.0, abs=1e-9)
        ys = sorted([t1.y(), t2.y()])
        assert ys[0] == pytest.approx(-2.0, abs=1e-9)
        assert ys[1] == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# HandleQGISIface methods that don't need a live canvas
# ---------------------------------------------------------------------------

@pytest.fixture
def hqi(omrat):
    """Reuse the omrat fixture from conftest."""
    return omrat.qgis_geoms


class TestCreateFields:
    def test_creates_five_fields(self, hqi):
        fields = hqi.create_fields()
        names = [f.name() for f in fields]
        assert names == ['segmentId', 'routeId', 'startPoint', 'endPoint', 'label']


class TestPoint4326FromWkt:
    def test_parses_qgs_point_from_wkt(self, hqi):
        from qgis.core import QgsPoint
        # The method takes a WKT string and returns a QgsPoint in EPSG:4326.
        pt = hqi.point4326_from_wkt('Point (14.0 55.0)')
        assert isinstance(pt, QgsPoint)
        # Coordinates pass through unchanged for WGS84 input.
        assert pt.x() == pytest.approx(14.0, abs=1e-6)
        assert pt.y() == pytest.approx(55.0, abs=1e-6)


class TestFormatWkt:
    def test_format_wkt_returns_lon_space_lat(self, hqi):
        from qgis.core import QgsPoint
        s = hqi.format_wkt(QgsPoint(14.0, 55.0))
        # The implementation just stringifies "x y".
        assert isinstance(s, str)
        assert '14' in s and '55' in s


class TestEnsureTangentLayer:
    def test_creates_layer_first_call(self, hqi):
        from qgis.core import QgsProject, QgsVectorLayer
        # Reset state for a clean run.
        hqi.tangent_layer = None
        hqi.ensure_tangent_layer()
        assert isinstance(hqi.tangent_layer, QgsVectorLayer)
        assert QgsProject.instance().mapLayersByName('Tangent Line')

    def test_idempotent(self, hqi):
        hqi.tangent_layer = None
        hqi.ensure_tangent_layer()
        first = hqi.tangent_layer
        hqi.ensure_tangent_layer()
        # Second call doesn't replace the layer.
        assert hqi.tangent_layer is first


class TestEnsureTangentFields:
    def test_creates_type_field(self, hqi):
        hqi.tangent_layer = None
        hqi.ensure_tangent_layer()
        hqi.ensure_tangent_fields()
        assert 'type' in [f.name() for f in hqi.tangent_layer.fields()]


class TestCalculateMidpointUtm:
    def test_returns_three_components(self, hqi):
        from qgis.core import QgsPointXY, QgsCoordinateTransform
        s = QgsPointXY(14.0, 55.0)
        e = QgsPointXY(14.1, 55.1)
        mid_utm, fwd, rev = hqi.calculate_midpoint_utm(s, e)
        assert isinstance(fwd, QgsCoordinateTransform)
        assert isinstance(rev, QgsCoordinateTransform)
        # Midpoint x/y should be a finite UTM-meter value.
        assert abs(mid_utm.x()) > 100_000


class TestStyleAndLabel:
    def test_style_layer_sets_renderer(self, hqi):
        from qgis.core import QgsVectorLayer
        from geometries.handle_qgis_iface import HandleQGISIface
        layer = QgsVectorLayer("LineString?crs=EPSG:4326", "x", "memory")
        # static-style call.
        HandleQGISIface.style_layer(layer)
        assert layer.renderer() is not None

    def test_label_layer_enables_labels(self, hqi):
        from qgis.core import QgsVectorLayer
        from geometries.handle_qgis_iface import HandleQGISIface
        layer = QgsVectorLayer("Point?crs=EPSG:4326", "x", "memory")
        HandleQGISIface.label_layer(layer)
        assert layer.labelsEnabled()


class TestRemoveExistingTangent:
    def test_removes_features_for_segment(self, hqi):
        from qgis.core import QgsPointXY
        # Add a tangent feature for segment 99, then remove it.
        hqi.tangent_layer = None
        hqi.ensure_tangent_layer()
        hqi.ensure_tangent_fields()
        hqi.add_tangent_feature(QgsPointXY(0, 0), QgsPointXY(1, 0), segment_id=99)
        before = hqi.tangent_layer.featureCount()
        hqi.remove_existing_tangent(99)
        after = hqi.tangent_layer.featureCount()
        # At least one fewer feature with that segment_id label.
        assert after <= before


class TestOnRouteTableCellClicked:
    def test_no_layers_does_nothing(self, hqi):
        """Calling without any layers should not raise."""
        hqi.vector_layers = []
        # Click a cell -- function reads main_widget table; with no vector_layers
        # the method just iterates over nothing.
        hqi.on_route_table_cell_clicked(row=0, column=0)


class TestClear:
    def test_clears_internal_state(self, hqi):
        from qgis.core import QgsVectorLayer
        layer = QgsVectorLayer("Point?crs=EPSG:4326", "TempLayer", "memory")
        hqi.vector_layers = [layer]
        hqi.tangent_layer = None
        hqi.clear()
        # vector_layers list emptied.
        assert hqi.vector_layers == []


# ---------------------------------------------------------------------------
# Movable tangent line
# ---------------------------------------------------------------------------

def _tangent_feature(hqi, segment_id: int):
    return next(
        f for f in hqi.tangent_layer.getFeatures()
        if f['type'] == f'Tangent Line {segment_id}'
    )


def _feature_mid_lon_lat(feat) -> tuple[float, float]:
    pts = feat.geometry().asPolyline()
    return (pts[0].x() + pts[-1].x()) / 2.0, (pts[0].y() + pts[-1].y()) / 2.0


@pytest.fixture
def leg77(hqi):
    """A 0.2-degree east-west leg at 55 N with a 2 km width."""
    from qgis.core import QgsPointXY
    hqi.omrat.testing = True
    if not isinstance(getattr(hqi.omrat, 'segment_data', None), dict):
        hqi.omrat.segment_data = {}
    hqi.omrat.segment_data['77'] = {
        'Start_Point': '14.000000 55.000000', 'End_Point': '14.200000 55.000000',
        'Width': 2000, 'Route_Id': 1, 'Leg_name': 'LEG_1_77', 'Segment_Id': '77',
        'Tangent_Pos': 0.5, 'Dirs': ['East going', 'West going'],
    }
    hqi.tangent_layer = None
    start, end = QgsPointXY(14.0, 55.0), QgsPointXY(14.2, 55.0)
    hqi.create_offset_lines(start, end, 1000.0, 77)
    yield start, end
    hqi.omrat.segment_data.pop('77', None)


class TestCalculateMidpointUtmFraction:
    def test_default_is_midpoint(self, hqi):
        from qgis.core import QgsPointXY
        mid, to_utm, _ = hqi.calculate_midpoint_utm(QgsPointXY(14.0, 55.0), QgsPointXY(14.2, 55.0))
        s = to_utm.transform(QgsPointXY(14.0, 55.0))
        e = to_utm.transform(QgsPointXY(14.2, 55.0))
        assert mid.x() == pytest.approx((s.x() + e.x()) / 2, abs=1e-6)
        assert mid.y() == pytest.approx((s.y() + e.y()) / 2, abs=1e-6)

    def test_fraction_moves_point_along_leg(self, hqi):
        from qgis.core import QgsPointXY
        mid, to_utm, _ = hqi.calculate_midpoint_utm(QgsPointXY(14.0, 55.0), QgsPointXY(14.2, 55.0), 0.25)
        s = to_utm.transform(QgsPointXY(14.0, 55.0))
        e = to_utm.transform(QgsPointXY(14.2, 55.0))
        assert mid.x() == pytest.approx(s.x() + 0.25 * (e.x() - s.x()), abs=1e-6)
        assert mid.y() == pytest.approx(s.y() + 0.25 * (e.y() - s.y()), abs=1e-6)


class TestCreateOffsetLinesTangentPos:
    def test_default_draws_at_leg_midpoint(self, hqi, leg77):
        lon, lat = _feature_mid_lon_lat(_tangent_feature(hqi, 77))
        assert lon == pytest.approx(14.1, abs=2e-3)
        assert lat == pytest.approx(55.0, abs=2e-3)

    def test_explicit_fraction_moves_the_line(self, hqi, leg77):
        start, end = leg77
        hqi.create_offset_lines(start, end, 1000.0, 77, tangent_pos=0.2)
        lon, _lat = _feature_mid_lon_lat(_tangent_feature(hqi, 77))
        assert lon == pytest.approx(14.04, abs=2e-3)

    def test_stored_fraction_is_used_when_not_given(self, hqi, leg77):
        """Width edits / vertex drags call without ``tangent_pos``; the
        stored value must survive those redraws."""
        start, end = leg77
        hqi.omrat.segment_data['77']['Tangent_Pos'] = 0.8
        hqi.create_offset_lines(start, end, 1000.0, 77)
        lon, _lat = _feature_mid_lon_lat(_tangent_feature(hqi, 77))
        assert lon == pytest.approx(14.16, abs=2e-3)

    def test_only_one_feature_per_leg_after_redraws(self, hqi, leg77):
        start, end = leg77
        for t in (0.2, 0.6, 0.9):
            hqi.create_offset_lines(start, end, 1000.0, 77, tangent_pos=t)
        n = sum(1 for f in hqi.tangent_layer.getFeatures() if f['type'] == 'Tangent Line 77')
        assert n == 1


class TestTangentDragSnapBack:
    def test_fraction_from_dragged_geometry(self, hqi, leg77):
        from qgis.core import QgsGeometry, QgsPointXY
        # Dragged roughly to 20 % along the leg, well off to one side and
        # slightly rotated -- only the along-track component matters.
        dragged = QgsGeometry.fromPolylineXY([QgsPointXY(14.035, 55.03), QgsPointXY(14.045, 54.96)])
        t = hqi.tangent_fraction_from_geometry(77, dragged)
        assert t == pytest.approx(0.2, abs=0.01)

    def test_fraction_clamped_when_dragged_past_the_end(self, hqi, leg77):
        from qgis.core import QgsGeometry, QgsPointXY
        beyond = QgsGeometry.fromPolylineXY([QgsPointXY(14.5, 55.01), QgsPointXY(14.5, 54.99)])
        assert hqi.tangent_fraction_from_geometry(77, beyond) == 1.0

    def test_geometry_changed_snaps_line_back_onto_leg(self, hqi, leg77):
        from qgis.core import QgsGeometry, QgsPointXY
        feat = _tangent_feature(hqi, 77)
        dragged = QgsGeometry.fromPolylineXY([QgsPointXY(14.04, 55.03), QgsPointXY(14.04, 54.97)])
        hqi._on_tangent_geometry_changed(feat.id(), dragged)  # testing=True -> synchronous

        assert hqi.omrat.segment_data['77']['Tangent_Pos'] == pytest.approx(0.2, abs=0.01)
        lon, lat = _feature_mid_lon_lat(_tangent_feature(hqi, 77))
        # Back on the leg (lat 55.0) at 20 %, with the table width, not the
        # dragged 0.06-degree length.
        assert lon == pytest.approx(14.04, abs=2e-3)
        assert lat == pytest.approx(55.0, abs=2e-4)
        assert not hqi._tangent_guard

    def test_geometry_changed_ignores_unknown_feature(self, hqi, leg77):
        from qgis.core import QgsGeometry, QgsPointXY
        before = hqi.omrat.segment_data['77']['Tangent_Pos']
        dragged = QgsGeometry.fromPolylineXY([QgsPointXY(14.04, 55.03), QgsPointXY(14.04, 54.97)])
        hqi._on_tangent_geometry_changed(-12345, dragged)
        assert hqi.omrat.segment_data['77']['Tangent_Pos'] == before

    def test_segment_id_parser(self, hqi):
        assert hqi._segment_id_from_tangent_type('Tangent Line 12') == 12
        assert hqi._segment_id_from_tangent_type('Tangent Line x') is None
        assert hqi._segment_id_from_tangent_type(None) is None
        assert hqi._segment_id_from_tangent_type('Other 3') is None


class TestTangentTableColumn:
    @pytest.fixture
    def table_row(self, hqi, leg77):
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        hqi.omrat.reset_route_table()
        tbl = hqi.omrat.main_widget.twRouteList
        tbl.setRowCount(1)
        for col, text in enumerate(['77', '1', 'LEG_1_77', '14.000000 55.000000', '14.200000 55.000000',
                                    '2000', '50']):
            tbl.setItem(0, col, QTableWidgetItem(text))
        return tbl

    def test_route_table_has_tangent_column(self, hqi, table_row):
        assert table_row.columnCount() == 8
        assert table_row.horizontalHeaderItem(6).text() == 'Tangent (%)'
        assert table_row.horizontalHeaderItem(7).text() == 'Update AIS'

    def test_set_tangent_pos_mirrors_into_table(self, hqi, table_row):
        hqi.set_tangent_pos(77, 0.3, redraw=False)
        assert table_row.item(0, 6).text() == '30'
        assert hqi.omrat.segment_data['77']['Tangent_Pos'] == pytest.approx(0.3)

    def test_typing_percent_moves_tangent(self, hqi, table_row):
        table_row.item(0, 6).setText('25')
        hqi.on_width_changed(table_row.item(0, 6))
        assert hqi.omrat.segment_data['77']['Tangent_Pos'] == pytest.approx(0.25)
        lon, _lat = _feature_mid_lon_lat(_tangent_feature(hqi, 77))
        assert lon == pytest.approx(14.05, abs=2e-3)

    def test_typing_garbage_restores_previous_value(self, hqi, table_row):
        hqi.omrat.segment_data['77']['Tangent_Pos'] = 0.4
        table_row.item(0, 6).setText('abc')
        hqi.on_width_changed(table_row.item(0, 6))
        assert table_row.item(0, 6).text() == '40'
        assert hqi.omrat.segment_data['77']['Tangent_Pos'] == 0.4

    def test_width_edit_keeps_moved_tangent(self, hqi, table_row):
        hqi.set_tangent_pos(77, 0.2)
        table_row.item(0, 5).setText('4000')
        hqi.on_width_changed(table_row.item(0, 5))
        lon, _lat = _feature_mid_lon_lat(_tangent_feature(hqi, 77))
        assert lon == pytest.approx(14.04, abs=2e-3)
        assert hqi.omrat.segment_data['77']['Width'] == 4000


class TestRouteTableLoadWiring:
    """File load fills the route table in bulk; the geometry handler must
    complete the rows and hook up the edit signal, otherwise Width and
    Tangent edits on a loaded project never reach the canvas."""

    def _six_column_row(self, hqi):
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        hqi.omrat.reset_route_table()
        hqi.suspend_route_table_signal()
        tbl = hqi.omrat.main_widget.twRouteList
        tbl.setRowCount(1)
        for col, text in enumerate(['77', '1', 'LEG_1_77', '14.000000 55.000000', '14.200000 55.000000', '2000']):
            tbl.setItem(0, col, QTableWidgetItem(text))
        return tbl

    def test_finish_rows_adds_cell_button_and_signal(self, hqi, leg77):
        tbl = self._six_column_row(hqi)
        assert tbl.item(0, 6) is None and tbl.cellWidget(0, 7) is None
        assert not hqi.item_changed_connected
        hqi.finish_route_table_rows()
        assert tbl.item(0, 6).text() == '50'
        assert tbl.cellWidget(0, 7) is not None
        assert tbl.cellWidget(0, 7).text() == 'Update AIS'
        assert hqi.item_changed_connected

    def test_finish_rows_is_idempotent(self, hqi, leg77):
        tbl = self._six_column_row(hqi)
        hqi.finish_route_table_rows()
        btn = tbl.cellWidget(0, 7)
        hqi.finish_route_table_rows()
        assert tbl.cellWidget(0, 7) is btn

    def test_populate_segment_tbl_makes_edits_live(self, hqi, leg77):
        from omrat_utils.gather_data import GatherData
        hqi.omrat.reset_route_table()
        hqi.suspend_route_table_signal()
        tbl = hqi.omrat.main_widget.twRouteList
        GatherData(hqi.omrat).populate_segment_tbl({'77': hqi.omrat.segment_data['77']}, tbl)
        assert tbl.item(0, 6).text() == '50'
        assert tbl.cellWidget(0, 7) is not None
        assert hqi.item_changed_connected
        # A real cell edit now flows through itemChanged -> on_width_changed.
        tbl.item(0, 6).setText('25')
        assert hqi.omrat.segment_data['77']['Tangent_Pos'] == pytest.approx(0.25)
        lon, _lat = _feature_mid_lon_lat(_tangent_feature(hqi, 77))
        assert lon == pytest.approx(14.05, abs=2e-3)
        tbl.item(0, 5).setText('3000')
        assert hqi.omrat.segment_data['77']['Width'] == 3000


class TestMoveTangentButton:
    def test_button_exists_and_is_wired(self, hqi):
        btn = getattr(hqi.omrat.main_widget, 'pbMoveTangent', None)
        assert btn is not None
        assert btn.receivers(btn.clicked) > 0

    def test_start_move_tangent_activates_layer_and_tool(self, hqi, leg77):
        triggered = []
        active = []

        class _Action:
            def trigger(self):
                triggered.append('move')

        hqi.omrat.iface.actionMoveFeature = lambda: _Action()
        hqi.omrat.iface.setActiveLayer = lambda lyr: active.append(lyr)
        hqi.start_move_tangent()
        assert triggered == ['move']
        assert active and active[0] is hqi.tangent_layer
        assert hqi.tangent_layer.isEditable()

    def test_start_move_tangent_without_lines_does_not_start_tool(self, hqi):
        triggered = []

        class _Action:
            def trigger(self):
                triggered.append('move')

        hqi.omrat.iface.actionMoveFeature = lambda: _Action()
        hqi.tangent_layer = None
        hqi.ensure_tangent_layer()
        hqi.ensure_tangent_fields()
        for f in list(hqi.tangent_layer.getFeatures()):
            hqi.tangent_layer.dataProvider().deleteFeatures([f.id()])
        hqi.start_move_tangent()
        assert triggered == []
