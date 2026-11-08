"""QGIS tests for **Depths > Simplify...**: the headless cores (replace the
depth table and layer, keep the original, restore) and the modeless dialog
(suggested limit and corridor length, zone layer, clean close).  Needs the
QGIS conftest."""
from __future__ import annotations

import pytest
from shapely.geometry import box

LOGNORMAL = {'use_lognormal': True, 'std': 0.97, 'loc': 0.2, 'scale': 0.5, 'func': ''}

# One E-W leg at 56 N with ships in the drawn direction; the structure is 3.3 km north of it.
LEG = {'Start_Point': '11.000000 56.000000', 'End_Point': '11.100000 56.000000', 'line_length': 6200.0,
       'Leg_name': 'LEG_1_1', 'Dirs': ['East going', 'West going'],
       'weight1_1': 100, 'mean1_1': 0.0, 'std1_1': 500.0, 'weight2_1': 100, 'mean2_1': 0.0, 'std2_1': 500.0}
STRUCTURE = box(11.045, 56.030, 11.055, 56.036).wkt
DEPTHS = [
    ['1', '0.0', box(10.90, 55.80, 10.95, 55.85).wkt],
    ['2', '12.0', box(11.15, 56.10, 11.20, 56.15).wkt],
    ['3', '18.0', box(11.20, 56.20, 11.25, 56.25).wkt],
    ['4', '21.0', box(11.04, 56.010, 11.06, 56.020).wkt],   # deep, between the leg and the structure
    ['5', '30.0', box(12.00, 57.00, 12.10, 57.10).wkt],     # deep, far away
    ['6', '24.0', box(11.04, 55.900, 11.06, 55.920).wkt],   # deep, south of the leg: no drift reaches the farm
]


@pytest.fixture
def project(omrat):
    from qgis.PyQt.QtWidgets import QTableWidgetItem
    om = omrat
    om.testing = True
    om.object.replace_depths([list(r) for r in DEPTHS])
    tbl = om.main_widget.twObjectList
    tbl.setRowCount(1)
    for col, value in enumerate(['1', '0', STRUCTURE]):
        tbl.setItem(0, col, QTableWidgetItem(value))
    om.segment_data = {'1': dict(LEG)}
    om.drift_values = {'speed': 1.0, 'repair': dict(LOGNORMAL),
                       'rose': {str(c): 0.125 for c in range(0, 360, 45)}}
    om.traffic_data = {'1': {
        'East going': {'Frequency (ships/year)': [[10.0]], 'Draught (meters)': [[16.1]]},
        'West going': {'Frequency (ships/year)': [[0.0]], 'Draught (meters)': [[0.0]]},
    }}
    om._depth_simplify_state = None
    yield om
    from omrat_utils.depth_simplify_dialog import BEFORE_LAYER_NAME, ZONE_LAYER_NAME, remove_layer
    dlg = getattr(om, '_depth_simplify_dlg', None)
    if dlg is not None:
        dlg.close()
        om._depth_simplify_dlg = None
    remove_layer(ZONE_LAYER_NAME)
    remove_layer(BEFORE_LAYER_NAME)


def _layer(name):
    from qgis.core import QgsProject
    layers = QgsProject.instance().mapLayersByName(name)
    return layers[0] if layers else None


def _depth_classes(om):
    return [r[1] for r in om.object.depth_rows()]


def _model_reach(om):
    from geometries.depth_simplify import corridor_length_m
    return corridor_length_m(om.drift_values, om.segment_data)


class TestHeadless:
    def test_apply_keeps_detail_and_corridors(self, project):
        from omrat_utils.depth_simplify_dialog import BEFORE_LAYER_NAME, apply_simplification
        om = project
        summary = apply_simplification(om, 18.0, _model_reach(om))
        assert _depth_classes(om) == ['0.0', '12.0', '18.0', '21.0']
        assert (summary['kept'], summary['clipped'], summary['dropped']) == (3, 1, 2)
        assert summary['legs'] == 1 and summary['corridors'] >= 1 and summary['zone_km2'] > 0
        assert om.object.depth_layer.featureCount() == 4
        before = _layer(BEFORE_LAYER_NAME)
        assert before is not None and before.featureCount() == 6
        from qgis.core import QgsProject
        node = QgsProject.instance().layerTreeRoot().findLayer(before.id())
        assert not node.itemVisibilityChecked()

    def test_second_run_starts_from_the_original(self, project):
        from omrat_utils.depth_simplify_dialog import BEFORE_LAYER_NAME, apply_simplification
        om = project
        apply_simplification(om, 18.0, 100.0)          # corridors too short to reach: class 21 goes too
        assert _depth_classes(om) == ['0.0', '12.0', '18.0']
        apply_simplification(om, 30.0, _model_reach(om))
        assert _depth_classes(om) == ['0.0', '12.0', '18.0', '21.0', '30.0', '24.0']
        assert _layer(BEFORE_LAYER_NAME).featureCount() == 6

    def test_restore_original(self, project):
        from omrat_utils.depth_simplify_dialog import (
            BEFORE_LAYER_NAME, apply_simplification, is_simplified, restore_original)
        om = project
        apply_simplification(om, 12.0, 100.0)
        assert is_simplified(om)
        assert restore_original(om)
        assert om.object.depth_rows() == DEPTHS
        assert om.object.depth_layer.featureCount() == 6
        assert _layer(BEFORE_LAYER_NAME) is None
        assert not restore_original(om)

    def test_no_restore_after_the_depths_changed(self, project):
        from omrat_utils.depth_simplify_dialog import apply_simplification, restore_original, source_rows
        om = project
        apply_simplification(om, 12.0, 100.0)
        om.object.replace_depths([['1', '3.0', box(0, 0, 1, 1).wkt]])
        assert not restore_original(om)
        assert source_rows(om) == om.object.depth_rows()

    def test_without_depths(self, project):
        from omrat_utils.depth_simplify_dialog import apply_simplification
        project.object.replace_depths([])
        with pytest.raises(ValueError):
            apply_simplification(project, 12.0, 1000.0)


class TestDialog:
    def test_suggestions_zone_and_simplify(self, project):
        from geometries.depth_simplify import NM_M
        from omrat_utils.depth_simplify_dialog import ZONE_LAYER_NAME, open_depth_simplify
        om = project
        dlg = open_depth_simplify(om)
        assert dlg is not None and om._depth_simplify_dlg is dlg
        assert dlg.detail_limit() == 18.0               # first class deeper than 16.1 m
        assert dlg.sbLength.value() == pytest.approx(_model_reach(om) / NM_M, abs=0.05)
        assert _layer(ZONE_LAYER_NAME) is not None
        assert 'leg(s)' in dlg.lblZoneHint.text()
        assert not dlg.pbRestore.isEnabled()

        summary = dlg.on_simplify()
        assert summary is not None
        assert _depth_classes(om) == ['0.0', '12.0', '18.0', '21.0']
        assert dlg.pbRestore.isEnabled()
        assert 'Depth classes 6 -> 4' in dlg.lblResult.text()

        assert dlg.on_restore()
        assert om.object.depth_rows() == DEPTHS
        dlg.close()
        assert _layer(ZONE_LAYER_NAME) is None

    def test_editing_the_length_switches_to_custom(self, project):
        from omrat_utils.depth_simplify_dialog import DepthSimplifyDialog, open_depth_simplify
        dlg = open_depth_simplify(project)
        dlg.sbLength.setValue(1.0)
        assert dlg.cbQuantile.currentData() == DepthSimplifyDialog.CUSTOM
        assert dlg.length_m() == pytest.approx(1852.0)
        dlg.refresh_zone()                               # the timer would do it after a pause
        assert 'reach' in dlg.lblZoneHint.text()
        dlg.cbQuantile.setCurrentIndex(2)               # 99.99 %: 18.6 nm
        assert dlg.sbLength.value() == pytest.approx(18.6, abs=0.1)

    def test_no_corridor_reaches_a_structure(self, project):
        from omrat_utils.depth_simplify_dialog import open_depth_simplify
        project.traffic_data['1']['East going']['Frequency (ships/year)'] = [[0.0]]
        dlg = open_depth_simplify(project)
        assert 'No drift corridor reaches' in dlg.lblZoneHint.text()

    def test_reopen_raises_the_same_dialog(self, project):
        from omrat_utils.depth_simplify_dialog import open_depth_simplify
        first = open_depth_simplify(project)
        assert open_depth_simplify(project) is first

    def test_button_on_depths_tab(self, project):
        assert project.main_widget.pbSimplifyDepth.text() == 'Simplify...'
