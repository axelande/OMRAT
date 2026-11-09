"""QGIS tests for **Settings > Speed restriction zones...**: the zones
handler (map layer, style, vertex edits), save / load, Run model's
refusal of overlapping zones, the Traffic tab tag and the dialog.  Needs
the QGIS conftest."""
from __future__ import annotations

import copy
import json

import pytest


def _dist_keys() -> dict:
    d = {}
    for j in (1, 2):
        for i in (1, 2, 3):
            d[f'mean{j}_{i}'] = 0.0
            d[f'std{j}_{i}'] = 200.0 if i == 1 else 0.0
            d[f'weight{j}_{i}'] = 100.0 if i == 1 else 0.0
        d[f'u_min{j}'] = 0.0
        d[f'u_max{j}'] = 0.0
        d[f'u_p{j}'] = 0
        d[f'ai{j}'] = 180
    return d


SEG = {
    '77': {'Start_Point': '14.000000 55.000000', 'End_Point': '14.200000 55.000000',
           'Width': 5000, 'Route_Id': 1, 'Leg_name': 'LEG_1_77', 'Segment_Id': '77',
           'Dirs': ['East going', 'West going'], 'line_length': 12000.0, 'Tangent_Pos': 0.5, **_dist_keys()},
    '78': {'Start_Point': '14.200000 55.000000', 'End_Point': '14.400000 55.000000',
           'Width': 5000, 'Route_Id': 1, 'Leg_name': 'LEG_1_78', 'Segment_Id': '78',
           'Dirs': ['East going', 'West going'], 'line_length': 12000.0, 'Tangent_Pos': 0.5, **_dist_keys()},
}


def _box(x0, y0, x1, y1):
    return f'Polygon (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))'


AROUND_JUNCTION = _box(14.15, 54.95, 14.25, 55.05)
OVERLAPPING = _box(14.2, 54.95, 14.3, 55.05)
ELSEWHERE = _box(15.0, 56.0, 15.1, 56.1)


@pytest.fixture
def legs(omrat):
    omrat.testing = True
    omrat.segment_data = copy.deepcopy(SEG)
    omrat.load_lines({'segment_data': omrat.segment_data})
    yield omrat
    dlg = getattr(omrat, '_speed_restriction_dlg', None)
    if dlg is not None:
        dlg.close()


def _features(layer):
    return {int(f['id']): f for f in layer.getFeatures()}


# ---------------------------------------------------------------------------
# Handler + layer
# ---------------------------------------------------------------------------

def test_layer_follows_the_zones(legs):
    from qgis.core import QgsProject, QgsRuleBasedRenderer
    from omrat_utils.speed_restriction_zones import DISABLED_RULE_LABEL, ENABLED_RULE_LABEL, LAYER_NAME
    zones = legs.speed_zones
    zone = zones.add_zone(AROUND_JUNCTION, name='Junction', limit_kn=8.0)
    layer = zones.layer
    assert layer is not None and layer.name() == LAYER_NAME
    assert QgsProject.instance().mapLayer(layer.id()) is not None
    renderer = layer.renderer()
    assert isinstance(renderer, QgsRuleBasedRenderer)
    assert [r.label() for r in renderer.rootRule().children()] == [ENABLED_RULE_LABEL, DISABLED_RULE_LABEL]
    feat = _features(layer)[zone['id']]
    assert feat['limit_kn'] == 8.0 and feat['enabled'] == 1 and feat['name'] == 'Junction'
    zones.update_zone(zone['id'], enabled=False, limit_kn=6.0)
    feat = _features(layer)[zone['id']]
    assert feat['enabled'] == 0 and feat['limit_kn'] == 6.0
    zones.remove_zone(zone['id'])
    assert zones.layer is None
    assert not QgsProject.instance().mapLayersByName(LAYER_NAME)


def test_vertex_edit_reaches_the_zone(legs):
    from qgis.core import QgsGeometry
    zones = legs.speed_zones
    zone = zones.add_zone(AROUND_JUNCTION)
    layer = zones.layer
    fid = next(layer.getFeatures()).id()
    layer.startEditing()
    layer.changeGeometry(fid, QgsGeometry.fromWkt(_box(14.1, 54.9, 14.25, 55.05)))
    assert zones.zone(zone['id'])['wkt'].replace(' ', '').startswith('Polygon((14.1')
    layer.rollBack()


def test_overlap_is_reported_and_blocks_the_run(legs):
    from omrat_utils.gather_data import GatherData
    zones = legs.speed_zones
    zones.add_zone(AROUND_JUNCTION)
    assert zones.last_messages == []
    second = zones.add_zone(OVERLAPPING)
    assert any('cannot overlap' in m for m in zones.last_messages)
    assert set(zones.problems()) == {1, second['id']}
    data = GatherData(legs).get_all_for_save()
    assert legs._speed_zones_ok_for_run(data) is False
    zones.update_zone(second['id'], enabled=False)
    data = GatherData(legs).get_all_for_save()
    assert legs._speed_zones_ok_for_run(data) is True


def test_traffic_tab_tag(legs):
    zones = legs.speed_zones
    assert zones.leg_tags('77') == []
    zones.add_zone(_box(14.05, 54.95, 14.1, 55.05))           # middle of leg 77 only
    assert zones.leg_tags('77') == ['speed zone'] and zones.leg_tags('78') == []
    zones.update_zone(1, enabled=False)
    assert zones.leg_tags('77') == []


def test_style_key_is_collected(legs):
    from omrat_utils.layer_styles import collect_styles
    legs.speed_zones.add_zone(AROUND_JUNCTION)
    assert 'speed_zones' in collect_styles(legs)


def test_load_from_layer_reads_limit_and_name(legs):
    from qgis.PyQt.QtCore import QMetaType
    from qgis.core import QgsFeature, QgsField, QgsGeometry, QgsVectorLayer
    src = QgsVectorLayer('Polygon?crs=EPSG:4326', 'src', 'memory')
    src.dataProvider().addAttributes([QgsField('NAMN', QMetaType.Type.QString),
                                      QgsField('speed_kn', QMetaType.Type.Double)])
    src.updateFields()
    feats = []
    for wkt, name, limit in ((AROUND_JUNCTION, 'A', 7.0), (ELSEWHERE, 'B', None)):
        f = QgsFeature(src.fields())
        f.setGeometry(QgsGeometry.fromWkt(wkt))
        f.setAttributes([name, limit])
        feats.append(f)
    src.dataProvider().addFeatures(feats)
    added = legs.speed_zones.add_from_layer(src, 'speed_kn', 'NAMN', default_limit=9.0)
    assert [(z['name'], z['limit_kn']) for z in added] == [('A', 7.0), ('B', 9.0)]


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(legs, tmp_path):
    from compute.speed_restriction import ZONES_KEY, new_rule
    from omrat_utils.storage import Storage
    zones = legs.speed_zones
    zone = zones.add_zone(AROUND_JUNCTION, name='Junction', limit_kn=8.0, compliance_pct=80.0)
    zones.update_zone(zone['id'], rules=[new_rule(ship_type=0, restricted=False)])
    target = tmp_path / 'zones.omrat'
    assert Storage(legs).store_all(str(target)) == str(target)
    saved = json.loads(target.read_text())
    assert saved[ZONES_KEY][0]['compliance_pct'] == 80.0
    assert saved[ZONES_KEY][0]['rules'][0]['restricted'] is False

    legs.clear_model()
    assert zones.zones == [] and zones.layer is None
    Storage(legs).load_from_path(str(target))
    assert zones.to_save() == saved[ZONES_KEY]
    assert zones.layer is not None
    legs.mark_project_saved()
    assert legs.has_unsaved_changes() is False


def test_project_without_zones_has_no_key(legs):
    from compute.speed_restriction import ZONES_KEY
    from omrat_utils.gather_data import GatherData
    assert ZONES_KEY not in GatherData(legs).get_all_for_save()


def test_merge_appends_with_new_ids(legs):
    from compute.speed_restriction import new_zone
    zones = legs.speed_zones
    zones.add_zone(AROUND_JUNCTION)
    zones.load([new_zone(1, ELSEWHERE, name='Other')], merge=True)
    assert [(z['id'], z['name']) for z in zones.zones] == [(1, 'Zone 1'), (2, 'Other')]


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

def _settle(qgis_app):
    qgis_app.processEvents()


def test_dialog_edits_zone_and_rules(legs, qgis_app):
    from omrat_utils.speed_restriction_dialog import (
        COL_APPLIES, COL_LEGS, COL_LIMIT, COL_ON, RCOL_LIMIT, RCOL_RESTRICTED,
    )
    from qgis.PyQt.QtCore import Qt
    zones = legs.speed_zones
    zone = zones.add_zone(AROUND_JUNCTION, limit_kn=8.0)
    legs.open_speed_restriction_zones()
    dlg = legs._speed_restriction_dlg
    assert dlg is not None and dlg.tw_zones.rowCount() == 1
    assert dlg.tw_zones.item(0, COL_LEGS).text() == '2'
    assert 'LEG_1_77' in dlg.tw_zones.item(0, COL_LEGS).toolTip()
    assert dlg.highlighted() >= 3                        # zone outline + two leg parts

    dlg.tw_zones.item(0, COL_LIMIT).setText('7,5')       # either decimal separator
    _settle(qgis_app)
    assert zones.zone(zone['id'])['limit_kn'] == 7.5
    dlg.tw_zones.item(0, COL_LIMIT).setText('abc')       # invalid: reverted
    _settle(qgis_app)
    assert zones.zone(zone['id'])['limit_kn'] == 7.5
    assert dlg.tw_zones.item(0, COL_LIMIT).text() == '7.5'

    dlg.add_rule()
    _settle(qgis_app)
    assert dlg.tw_rules.rowCount() == 1
    assert dlg.tw_zones.item(0, COL_APPLIES).text() == '1 rule'
    dlg.tw_rules.item(0, RCOL_RESTRICTED).setCheckState(Qt.CheckState.Unchecked)
    _settle(qgis_app)
    assert zones.zone(zone['id'])['rules'][0]['restricted'] is False
    dlg.tw_rules.item(0, RCOL_LIMIT).setText('6')
    _settle(qgis_app)
    assert zones.zone(zone['id'])['rules'][0]['limit_kn'] == 6.0
    dlg.add_rule()
    _settle(qgis_app)
    dlg.move_rule(-1, row=1)
    _settle(qgis_app)
    assert [r['limit_kn'] for r in zones.zone(zone['id'])['rules']] == [None, 6.0]
    dlg.remove_rule(row=0)
    _settle(qgis_app)
    assert len(zones.zone(zone['id'])['rules']) == 1

    dlg.tw_zones.item(0, COL_ON).setCheckState(Qt.CheckState.Unchecked)
    _settle(qgis_app)
    assert zones.zone(zone['id'])['enabled'] is False


def test_dialog_shows_overlap_and_cleans_up(legs, qgis_app):
    from omrat_utils.speed_restriction_dialog import COL_NAME
    zones = legs.speed_zones
    zones.add_zone(AROUND_JUNCTION)
    legs.open_speed_restriction_zones()
    dlg = legs._speed_restriction_dlg
    zones.add_zone(OVERLAPPING)
    _settle(qgis_app)
    assert dlg.tw_zones.rowCount() == 2
    assert 'cannot overlap' in dlg.tw_zones.item(1, COL_NAME).toolTip()
    dlg.select_zone(2)
    assert 'cannot overlap' in dlg.lbl_problems.text()
    assert 'Run model is blocked' in dlg.lbl_problems.text()
    dlg.close()
    _settle(qgis_app)
    assert legs._speed_restriction_dlg is None
    assert dlg.highlighted() == 0


def test_drawing_adds_a_zone(legs, qgis_app):
    from omrat_utils.speed_restriction_zones import DRAW_LAYER_NAME
    from qgis.core import QgsProject
    zones = legs.speed_zones
    legs.open_speed_restriction_zones()
    dlg = legs._speed_restriction_dlg
    dlg._on_draw()
    assert zones.is_drawing() and dlg.btn_draw.text() == 'Cancel drawing'
    assert QgsProject.instance().mapLayersByName(DRAW_LAYER_NAME)
    zones.finish_drawing(AROUND_JUNCTION)
    _settle(qgis_app)
    assert not zones.is_drawing()
    assert not QgsProject.instance().mapLayersByName(DRAW_LAYER_NAME)
    assert len(zones.zones) == 1 and dlg.tw_zones.rowCount() == 1
    assert dlg.btn_draw.text() == 'Draw new'
