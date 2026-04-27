"""Tests for ``OObject`` instance methods in ``omrat_utils.handle_object``.

The pure helpers are covered by ``test_handle_object_helpers.py``.  This
file exercises the class methods that need a real ``omrat`` fixture
(plugin + main_widget + iface mocks) provided by conftest.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def obj(omrat):
    return omrat.object


# ---------------------------------------------------------------------------
# _ensure_depth_layer / _apply_depth_graduated_style
# ---------------------------------------------------------------------------

class TestEnsureDepthLayer:
    def test_creates_layer_first_call(self, obj):
        from qgis.core import QgsVectorLayer
        layer = obj._ensure_depth_layer()
        assert isinstance(layer, QgsVectorLayer)
        # Has 'id' and 'depth' fields.
        names = [f.name() for f in layer.fields()]
        assert 'id' in names and 'depth' in names

    def test_idempotent(self, obj):
        l1 = obj._ensure_depth_layer()
        l2 = obj._ensure_depth_layer()
        assert l1 is l2


class TestApplyDepthGraduatedStyle:
    def test_no_depth_layer_returns_silently(self, obj):
        obj.depth_layer = None
        # Should not raise.
        obj._apply_depth_graduated_style()

    def test_with_features_applies_renderer(self, obj):
        # Create the layer + add a feature with valid depth.
        obj._add_depth_feature(
            depth_id=1, depth_value=5.0,
            wkt='POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            row=0, defer_style=False,
        )
        # After style application a renderer is set.
        assert obj.depth_layer.renderer() is not None

    def test_skips_when_no_finite_depths(self, obj):
        from qgis.core import QgsFeature, QgsField, QgsGeometry, QgsVectorLayer
        from qgis.PyQt.QtCore import QVariant
        # Create the layer.
        obj._ensure_depth_layer()
        # Don't add any features -> depths list is empty -> early return.
        obj._apply_depth_graduated_style()


# ---------------------------------------------------------------------------
# _add_depth_feature -- the public path used by load
# ---------------------------------------------------------------------------

class TestAddDepthFeature:
    def test_adds_feature_with_attributes(self, obj):
        obj._add_depth_feature(
            depth_id=42, depth_value=12.5,
            wkt='POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            row=3, defer_style=True,
        )
        feats = list(obj.depth_layer.getFeatures())
        assert len(feats) == 1
        f = feats[0]
        assert f.attribute('id') == 42
        assert f.attribute('depth') == 12.5
        # depth_feature_row maps fid -> row index.
        assert obj.depth_feature_row[f.id()] == 3


# ---------------------------------------------------------------------------
# _on_depth_geometry_changed
# ---------------------------------------------------------------------------

class TestOnDepthGeometryChanged:
    def test_writes_wkt_back_into_table(self, obj):
        from qgis.core import QgsGeometry
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        # Set up table with one row.
        tbl = obj.p.main_widget.twDepthList
        tbl.setRowCount(1)
        tbl.setColumnCount(3)
        tbl.setItem(0, 0, QTableWidgetItem('1'))
        tbl.setItem(0, 1, QTableWidgetItem('5.0'))
        tbl.setItem(0, 2, QTableWidgetItem('OLD'))
        # Add a depth feature; depth_feature_row[fid] = 0.
        obj._add_depth_feature(
            depth_id=1, depth_value=5.0,
            wkt='POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            row=0, defer_style=True,
        )
        fid = next(iter(obj.depth_feature_row))
        new_geom = QgsGeometry.fromWkt('POLYGON((10 10, 11 10, 11 11, 10 11, 10 10))')
        obj._on_depth_geometry_changed(fid, new_geom)
        # Cell at row 0, col 2 was updated.
        assert tbl.item(0, 2).text().startswith('Polygon ((10')

    def test_unknown_fid_is_noop(self, obj):
        from qgis.core import QgsGeometry
        # No features added; fid 999 unknown.
        obj._on_depth_geometry_changed(999, QgsGeometry.fromWkt('POINT(0 0)'))
        # Did not crash.


# ---------------------------------------------------------------------------
# _rebuild_depth_feature_row_map
# ---------------------------------------------------------------------------

class TestRebuildDepthFeatureRowMap:
    def test_no_layer_clears_map(self, obj):
        obj.depth_feature_row = {1: 0}
        obj.depth_layer = None
        obj._rebuild_depth_feature_row_map()
        assert obj.depth_feature_row == {}

    def test_matches_fid_attr_to_table_row(self, obj):
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        # Two depth features with ids 1, 2.
        obj._add_depth_feature(1, 5.0, 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))', 0, defer_style=True)
        obj._add_depth_feature(2, 7.0, 'POLYGON((2 2, 3 2, 3 3, 2 3, 2 2))', 1, defer_style=True)
        # Now reset the map and the table to mirror the ids in different rows.
        obj.depth_feature_row.clear()
        tbl = obj.p.main_widget.twDepthList
        tbl.setRowCount(2)
        tbl.setColumnCount(3)
        tbl.setItem(0, 0, QTableWidgetItem('2'))  # row 0 -> id 2
        tbl.setItem(1, 0, QTableWidgetItem('1'))  # row 1 -> id 1

        obj._rebuild_depth_feature_row_map()
        # Each fid mapped to the row whose id-cell text matches.
        for fid, row in obj.depth_feature_row.items():
            feat = obj.depth_layer.getFeature(fid)
            assert tbl.item(row, 0).text() == str(feat.attribute('id'))


# ---------------------------------------------------------------------------
# _select_file (file dialog)
# ---------------------------------------------------------------------------

class TestSelectFile:
    def test_returns_path_when_user_picks(self, obj, monkeypatch):
        import omrat_utils.handle_object as mod
        monkeypatch.setattr(
            mod.QFileDialog, 'getOpenFileName',
            lambda *a, **k: ('/tmp/x.shp', 'shp')
        )
        assert obj._select_file('Pick file') == '/tmp/x.shp'

    def test_returns_none_when_cancelled(self, obj, monkeypatch):
        import omrat_utils.handle_object as mod
        monkeypatch.setattr(
            mod.QFileDialog, 'getOpenFileName',
            lambda *a, **k: ('', '')
        )
        assert obj._select_file('Pick') is None


# ---------------------------------------------------------------------------
# add_simple_depth / store_depth round-trip
# ---------------------------------------------------------------------------

class TestSimpleDepthFlow:
    def test_add_then_store_populates_table(self, obj):
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        from qgis.core import QgsFeature, QgsGeometry
        # First click: button text 'Add manual' -> creates layer + flips to 'Save'.
        obj.p.main_widget.pbAddSimpleDepth.setText('Add manual')
        obj.add_simple_depth()
        assert obj.area is not None
        assert obj.p.main_widget.pbAddSimpleDepth.text() == 'Save'
        # Add a polygon feature to obj.area.
        prov = obj.area.dataProvider()
        feat = QgsFeature(obj.area.fields())
        feat.setGeometry(QgsGeometry.fromWkt('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'))
        prov.addFeature(feat)
        obj.area.updateExtents()
        # Second click: 'Save' branch -> store_depth -> populates table.
        obj.add_simple_depth()
        assert obj.p.main_widget.pbAddSimpleDepth.text() == 'Add manual'
        # twDepthList has 1 row after store_depth.
        assert obj.p.main_widget.twDepthList.rowCount() >= 1


# ---------------------------------------------------------------------------
# add_simple_object / store_object round-trip
# ---------------------------------------------------------------------------

class TestSimpleObjectFlow:
    def test_add_then_store_populates_table(self, obj):
        from qgis.core import QgsFeature, QgsGeometry
        obj.p.main_widget.pbAddSimpleObject.setText('Add manual')
        obj.add_simple_object()
        assert obj.area is not None
        assert obj.p.main_widget.pbAddSimpleObject.text() == 'Save'
        prov = obj.area.dataProvider()
        feat = QgsFeature(obj.area.fields())
        feat.setGeometry(QgsGeometry.fromWkt('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'))
        prov.addFeature(feat)
        obj.area.updateExtents()
        obj.add_simple_object()
        assert obj.p.main_widget.pbAddSimpleObject.text() == 'Add manual'
        assert obj.p.main_widget.twObjectList.rowCount() >= 1


# ---------------------------------------------------------------------------
# remove_depth / remove_object
# ---------------------------------------------------------------------------

class TestRemoveDepth:
    def test_no_selection_returns_silently(self, obj):
        # No selection in table -> early return.
        obj.remove_depth()

    def test_removes_selected_rows(self, obj):
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        # Add a depth feature & corresponding table row.
        obj._add_depth_feature(1, 5.0, 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))', 0, defer_style=True)
        tbl = obj.p.main_widget.twDepthList
        tbl.setRowCount(1); tbl.setColumnCount(3)
        tbl.setItem(0, 0, QTableWidgetItem('1'))
        tbl.setItem(0, 1, QTableWidgetItem('5'))
        tbl.setItem(0, 2, QTableWidgetItem('w'))
        # Select the row.
        tbl.selectRow(0)
        before = obj.depth_layer.featureCount()
        obj.remove_depth()
        after = obj.depth_layer.featureCount()
        assert after < before


class TestRemoveObject:
    def test_no_selection_returns_silently(self, obj):
        obj.remove_object()

    def test_removes_selected_row_and_layer(self, obj):
        from qgis.core import QgsVectorLayer, QgsProject
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        # Add a fake object layer + table row.
        layer = QgsVectorLayer("Polygon?crs=EPSG:4326", "TestObj", "memory")
        QgsProject.instance().addMapLayer(layer)
        obj.loaded_object_areas = [layer]
        tbl = obj.p.main_widget.twObjectList
        tbl.setRowCount(1); tbl.setColumnCount(3)
        tbl.setItem(0, 0, QTableWidgetItem('1'))
        tbl.setItem(0, 1, QTableWidgetItem('5'))
        tbl.setItem(0, 2, QTableWidgetItem('w'))
        tbl.selectRow(0)
        obj.remove_object()
        assert tbl.rowCount() == 0


# ---------------------------------------------------------------------------
# _cleanup_depth_layer / unload / clear
# ---------------------------------------------------------------------------

class TestCleanupAndUnload:
    def test_cleanup_with_no_layer_safe(self, obj):
        obj.depth_layer = None
        obj._depth_edit_buffer = None
        obj._cleanup_depth_layer()  # no raise

    def test_cleanup_removes_layer(self, obj):
        obj._add_depth_feature(1, 5.0, 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))', 0, defer_style=True)
        assert obj.depth_layer is not None
        obj._cleanup_depth_layer()
        assert obj.depth_layer is None
        assert obj.depth_feature_row == {}

    def test_unload(self, obj):
        # Should not raise even if some attrs are None.
        obj.area = None
        obj.depth_layer = None
        obj.unload()

    def test_clear(self, obj):
        obj.area = None
        obj.depth_layer = None
        obj.clear()


# ---------------------------------------------------------------------------
# load_objects / load_depths -- file dialog returns None
# ---------------------------------------------------------------------------

class TestLoadShortCircuits:
    def test_load_objects_no_file_returns_silently(self, obj, monkeypatch):
        import omrat_utils.handle_object as mod
        monkeypatch.setattr(
            mod.QFileDialog, 'getOpenFileName',
            lambda *a, **k: ('', '')
        )
        obj.load_objects()  # no raise

    def test_load_depths_no_file_returns_silently(self, obj, monkeypatch):
        import omrat_utils.handle_object as mod
        monkeypatch.setattr(
            mod.QFileDialog, 'getOpenFileName',
            lambda *a, **k: ('', '')
        )
        obj.load_depths()


# ---------------------------------------------------------------------------
# _load_layer / _populate_table
# ---------------------------------------------------------------------------

class TestPopulateTable:
    def test_populates_with_attribute(self, obj):
        from qgis.core import (
            QgsVectorLayer, QgsField, QgsFeature, QgsGeometry, QgsProject
        )
        from qgis.PyQt.QtCore import QVariant
        layer = QgsVectorLayer("Polygon?crs=EPSG:4326", "x", "memory")
        prov = layer.dataProvider()
        prov.addAttributes([QgsField("depth", QVariant.Double)])
        layer.updateFields()
        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsGeometry.fromWkt('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'))
        feat.setAttribute('depth', 7.5)
        prov.addFeature(feat)

        tbl = obj.p.main_widget.twDepthList
        tbl.setRowCount(0); tbl.setColumnCount(3)
        obj._populate_table(layer, tbl, 'depth')
        assert tbl.rowCount() == 1
        assert tbl.item(0, 1).text() == '7.5'


class TestLoadLayer:
    def test_invalid_returns_none(self, obj):
        result = obj._load_layer('/nonexistent/path.shp', 'NoLayer', [])
        assert result is None
