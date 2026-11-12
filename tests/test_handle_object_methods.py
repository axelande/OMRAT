"""Tests for ``OObject`` instance methods in ``omrat_utils.handle_object``.

The pure helpers are covered by ``test_handle_object_helpers.py``.  This
file exercises the class methods that need a real ``omrat`` fixture
(plugin + main_widget + iface mocks) provided by conftest.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
    def test_recreates_layer_removed_from_project(self, obj):
        """Layer deleted from the Layers panel must be re-added on next use."""
        from qgis.core import QgsProject
        l1 = obj._ensure_depth_layer()
        obj.depth_feature_row = {1: 0}
        QgsProject.instance().removeMapLayer(l1.id())
        l2 = obj._ensure_depth_layer()
        assert l2 is not l1
        assert QgsProject.instance().mapLayer(l2.id()) is l2
        assert obj.depth_layer is l2
        assert obj.depth_feature_row == {}

    def test_load_area_after_external_removal_adds_layer(self, obj):
        """Regression: loading a model after the depth layer vanished showed nothing."""
        from qgis.core import QgsProject
        obj.area_type = 'depth'
        first = obj._ensure_depth_layer()
        QgsProject.instance().removeMapLayer(first.id())
        obj.load_area('Depth - 10m', 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))', row=0,
                      value='10', value_field='Depth', defer_style=True)
        assert obj.depth_layer is not first
        assert QgsProject.instance().mapLayer(obj.depth_layer.id()) is not None
        assert obj.depth_layer.featureCount() == 1

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
        from qgis.core import QgsFeature, QgsGeometry
        # First click: button text 'Add manually' -> creates layer + flips to 'Save'.
        obj.p.main_widget.pbAddSimpleDepth.setText('Add manually')
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
        assert obj.p.main_widget.pbAddSimpleDepth.text() == 'Add manually'
        # twDepthList has 1 row after store_depth.
        assert obj.p.main_widget.twDepthList.rowCount() >= 1


# ---------------------------------------------------------------------------
# add_simple_object / store_object round-trip
# ---------------------------------------------------------------------------

class TestSimpleObjectFlow:
    def test_add_then_store_populates_table(self, obj):
        from qgis.core import QgsFeature, QgsGeometry
        obj.p.main_widget.pbAddSimpleObject.setText('Add manually')
        obj.add_simple_object()
        assert obj.area is not None
        assert obj.p.main_widget.pbAddSimpleObject.text() == 'Save'
        prov = obj.area.dataProvider()
        feat = QgsFeature(obj.area.fields())
        feat.setGeometry(QgsGeometry.fromWkt('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'))
        prov.addFeature(feat)
        obj.area.updateExtents()
        obj.add_simple_object()
        assert obj.p.main_widget.pbAddSimpleObject.text() == 'Add manually'
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
        tbl.setRowCount(1)
        tbl.setColumnCount(3)
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
        tbl.setRowCount(1)
        tbl.setColumnCount(3)
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
# load_objects / load_depths from a file
# ---------------------------------------------------------------------------

SQUARE = [[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01], [0.0, 0.0]]


def _geojson(path, features, crs=None):
    """Write a polygon GeoJSON (GDAL honours the legacy ``crs`` member)."""
    import json
    doc = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': props,
         'geometry': {'type': 'Polygon', 'coordinates': [ring]}}
        for props, ring in features]}
    if crs:
        doc['crs'] = {'type': 'name', 'properties': {'name': crs}}
    path.write_text(json.dumps(doc), encoding='utf-8')
    return str(path)


def _shifted(dx):
    return [[x + dx, y] for x, y in SQUARE]


class TestLoadFromFile:
    def test_structure_height_from_height_field_any_case(self, obj, tmp_path, monkeypatch):
        path = _geojson(tmp_path / 's.geojson', [({'Height': 25.0}, SQUARE), ({'Height': 12.5}, _shifted(0.02))])
        monkeypatch.setattr(obj, '_select_file', lambda title: path)
        obj.load_objects()
        rows = obj.object_rows()
        assert [(r[0], float(r[1])) for r in rows] == [('1', 25.0), ('2', 12.5)]
        # one layer per structure, as on project load, so rows map to layers
        assert [lyr.name() for lyr in obj.loaded_object_areas] == ['Structure - 25.0m', 'Structure - 12.5m']

    def test_legacy_object_field_still_read(self, obj, tmp_path, monkeypatch):
        path = _geojson(tmp_path / 's.geojson', [({'object': 8.0}, SQUARE)])
        monkeypatch.setattr(obj, '_select_file', lambda title: path)
        obj.load_objects()
        assert float(obj.object_rows()[0][1]) == 8.0

    def test_removing_a_loaded_row_removes_its_own_layer(self, obj, tmp_path, monkeypatch):
        from qgis.core import QgsProject
        path = _geojson(tmp_path / 's.geojson', [({'height': 1.0}, SQUARE), ({'height': 2.0}, _shifted(0.02))])
        monkeypatch.setattr(obj, '_select_file', lambda title: path)
        obj.load_objects()
        first_id, second_id = (lyr.id() for lyr in obj.loaded_object_areas)
        obj.p.main_widget.twObjectList.selectRow(0)
        obj.remove_object()
        assert QgsProject.instance().mapLayer(first_id) is None
        assert QgsProject.instance().mapLayer(second_id) is not None
        assert [lyr.id() for lyr in obj.loaded_object_areas] == [second_id]
        assert [float(r[1]) for r in obj.object_rows()] == [2.0]

    def test_reprojects_non_wgs84_file_to_4326(self, obj, tmp_path, monkeypatch):
        # Regression: a file in UTM landed in twObjectList with projected
        # metres, which then blew up the calculation's forward-transform check.
        import re
        ring = [[604000, 6199000], [604500, 6199000], [604500, 6199500], [604000, 6199500], [604000, 6199000]]
        path = _geojson(tmp_path / 'utm.geojson', [({'height': 5.0}, ring)], crs='urn:ogc:def:crs:EPSG::32640')
        monkeypatch.setattr(obj, '_select_file', lambda title: path)
        obj.load_objects()
        wkt = obj.object_rows()[0][2]
        nums = [float(n) for n in re.findall(r'-?\d+\.\d+|-?\d+', wkt)]
        assert nums, f'no coordinates parsed from {wkt!r}'
        assert all(-180.0 <= n <= 180.0 for n in nums), f'WKT still in projected CRS: {wkt!r}'

    def test_depth_field_any_case(self, obj, tmp_path, monkeypatch):
        path = _geojson(tmp_path / 'd.geojson', [({'DEPTH': 7.5}, SQUARE)])
        monkeypatch.setattr(obj, '_select_file', lambda title: path)
        obj.load_depths()
        assert float(obj.depth_rows()[0][1]) == 7.5


class TestManualAddAfterLoad:
    """Add manually used its own counter, which no load advanced: after a
    load of two rows the first manual entry cut the table down to one row."""

    @staticmethod
    def _draw(obj, kind):
        from qgis.core import QgsFeature, QgsGeometry
        w = obj.p.main_widget
        button, click = ((w.pbAddSimpleObject, obj.add_simple_object) if kind == 'object'
                         else (w.pbAddSimpleDepth, obj.add_simple_depth))
        button.setText('Add manually')
        click()
        feat = QgsFeature(obj.area.fields())
        feat.setGeometry(QgsGeometry.fromWkt('POLYGON((5 5, 6 5, 6 6, 5 6, 5 5))'))
        obj.area.dataProvider().addFeature(feat)
        click()
        assert button.text() == 'Add manually'

    def test_structure_is_appended(self, obj, tmp_path, monkeypatch):
        path = _geojson(tmp_path / 's.geojson', [({'height': 25.0}, SQUARE), ({'height': 12.5}, _shifted(0.02))])
        monkeypatch.setattr(obj, '_select_file', lambda title: path)
        obj.load_objects()
        self._draw(obj, 'object')
        rows = obj.object_rows()
        assert [r[0] for r in rows] == ['1', '2', '3']
        assert [float(r[1]) for r in rows[:2]] == [25.0, 12.5]
        assert 'POLYGON' in rows[2][2].upper()
        # the drawn structure gets the same one-feature layer as a loaded one
        assert [lyr.name() for lyr in obj.loaded_object_areas][-1] == 'Structure - 10.0m'
        assert len(obj.loaded_object_areas) == 3

    def test_depth_is_appended(self, obj, tmp_path, monkeypatch):
        path = _geojson(tmp_path / 'd.geojson', [({'depth': 3.0}, SQUARE), ({'depth': 6.0}, _shifted(0.02))])
        monkeypatch.setattr(obj, '_select_file', lambda title: path)
        obj.load_depths()
        self._draw(obj, 'depth')
        rows = obj.depth_rows()
        assert [r[0] for r in rows] == ['1', '2', '3']
        assert [float(r[1]) for r in rows[:2]] == [3.0, 6.0]


class TestDepthClassOnTheMap:
    """The depth layer shows a class by the value compute uses: an
    interval's shallower end (it showed the deeper end before 2026-11-12)."""

    @pytest.mark.parametrize('value, expected', [('10-20', 10.0), ('-5', -5.0), ('7.5', 7.5)])
    def test_project_load_colours_by_the_shallower_end(self, obj, value, expected):
        obj.area_type = 'depth'
        obj.load_area(f'Depth - {value}m', 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))', row=0,
                      value=value, value_field='Depth', defer_style=True)
        depths = [f['depth'] for f in obj.depth_layer.getFeatures()]
        assert depths == [expected]
