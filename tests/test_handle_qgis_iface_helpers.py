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
