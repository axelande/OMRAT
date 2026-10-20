"""QGIS tests for **Settings > Waypoint editor...**: the headless cores
(map line, model, junction registry) and the modeless dialog (list, map
layer, pick, detach button, clean close).  Needs the QGIS conftest."""
from __future__ import annotations

import copy

import pytest

A = (14.000000, 55.000000)
B = (14.000000, 55.200000)
C = (14.200000, 55.300000)
D = (13.800000, 55.300000)
E = (14.400000, 55.200000)


def _txt(xy):
    return f'{xy[0]:.6f} {xy[1]:.6f}'


def _make_leg_layer(hqi, seg_id: str, start: str, end: str):
    from qgis.core import QgsFeature, QgsGeometry, QgsPointXY, QgsProject, QgsVectorLayer
    vl = QgsVectorLayer("LineString?crs=EPSG:4326", f"LEG_1_{seg_id}", "memory")
    fields = hqi.create_fields()
    vl.dataProvider().addAttributes(fields.toList())
    vl.updateFields()
    feat = QgsFeature(vl.fields())
    (x1, y1), (x2, y2) = [tuple(float(v) for v in p.split()) for p in (start, end)]
    feat.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(x1, y1), QgsPointXY(x2, y2)]))
    feat.setAttributes([seg_id, 1, start, end, f"LEG_1_{seg_id}"])
    vl.dataProvider().addFeature(feat)
    hqi.style_layer(vl)
    QgsProject.instance().addMapLayer(vl)
    hqi.vector_layers.append(vl)
    return vl


def _line(layer):
    feat = next(layer.getFeatures())
    return [(p.x(), p.y()) for p in feat.geometry().asPolyline()]


def _close(a, b, tol=1e-6):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


@pytest.fixture
def hqi(omrat):
    return omrat.qgis_geoms


@pytest.fixture
def legs(hqi):
    """Legs 81, 82, 83 meet at B; leg 84 runs C -> E (not connected to B)."""
    from geometries.route_validation import haversine_m
    om = hqi.omrat
    om.testing = True
    base = {'Width': 2000, 'Route_Id': 1, 'Tangent_Pos': 0.5, 'Dirs': ['North going', 'South going']}
    ends = {'81': (A, B), '82': (B, C), '83': (B, D), '84': (C, E)}
    for k, (sp, ep) in ends.items():
        om.segment_data[k] = dict(copy.deepcopy(base), Segment_Id=k, Leg_name=f'LEG_1_{k}',
                                  Start_Point=_txt(sp), End_Point=_txt(ep), line_length=haversine_m(sp, ep))
    layers = {k: _make_leg_layer(hqi, k, _txt(sp), _txt(ep)) for k, (sp, ep) in ends.items()}
    om.junctions.rebuild_from_segments(om.segment_data, prefer_user=True)
    yield layers
    dlg = getattr(om, '_waypoint_editor_dlg', None)
    if dlg is not None:
        dlg.close()


def _wp_at(om, xy):
    from geometries.waypoints import find_waypoint_at
    from omrat_utils.waypoint_editor_dialog import sync_waypoints
    sync_waypoints(om)
    return find_waypoint_at(om.waypoints, xy)


def test_apply_detach_moves_the_leg_on_map_and_in_the_model(hqi, legs):
    from geometries.junctions import junction_id_for_point
    from geometries.waypoints import detach_limits
    from omrat_utils.waypoint_editor_dialog import apply_detach
    om = hqi.omrat
    b = _wp_at(om, B)
    _lo, gap, _hi = detach_limits(om.waypoints, om.segment_data, b, '82', 'start')
    result = apply_detach(om, b, '82', 'start', gap)

    new_wp = result['new_wp']
    seg = om.segment_data['82']
    assert seg['start_wp'] == new_wp
    assert seg['Start_Point'] == _txt(om.waypoints[new_wp])
    start, end = _line(legs['82'])
    assert _close(start, om.waypoints[new_wp]) and _close(end, C)
    # The legs that stayed did not move.
    assert _close(_line(legs['81'])[1], B) and _close(_line(legs['83'])[0], B)
    assert set(om.junctions.registry[junction_id_for_point(B)].legs) == {'81', '83'}
    assert seg['Dirs'] == ['North going', 'South going']


def test_apply_attach_extends_a_hand_edited_matrix(hqi, legs):
    from geometries.junctions import junction_id_for_point
    from omrat_utils.waypoint_editor_dialog import apply_attach
    om = hqi.omrat
    b = _wp_at(om, B)
    jid = junction_id_for_point(B)
    om.junctions.set_matrix(jid, {'81': {'82': 70, '83': 30}, '82': {'81': 100}, '83': {'81': 100}})
    result = apply_attach(om, '84', 'start', b)

    assert result['extended'] == [jid]
    assert _close(_line(legs['84'])[0], B)
    j = om.junctions.registry[jid]
    assert j.source == 'user'
    assert j.transitions['81']['82'] == pytest.approx(0.7)
    assert j.transitions['81']['84'] == 0.0
    assert sum(j.transitions['84'].values()) == pytest.approx(1.0)
    # C keeps leg 82, so its node is still there.
    assert _wp_at(om, C) is not None


def test_dialog_lists_waypoints_and_detaches_with_confirmation(hqi, legs, monkeypatch):
    from qgis.core import QgsProject
    from qgis.PyQt.QtCore import Qt
    from omrat_utils.waypoint_editor_dialog import WaypointEditorDialog, open_waypoint_editor
    om = hqi.omrat
    dlg = open_waypoint_editor(om)
    assert isinstance(dlg, WaypointEditorDialog)
    assert open_waypoint_editor(om) is dlg          # a second open raises the same one
    n_wps = dlg.lst_wp.count()
    assert n_wps == len(om.waypoints) == 5          # A B C D E
    assert QgsProject.instance().mapLayersByName(dlg.LAYER_NAME)

    b = _wp_at(om, B)
    assert dlg.select_waypoint(b, zoom=False)
    legs_on_b = [tuple(dlg.lst_legs.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(dlg.lst_legs.count())]
    assert sorted(legs_on_b) == [('81', 'end'), ('82', 'start'), ('83', 'start')]
    assert dlg.cmb_cand.count() == 2                # both ends of leg 84
    assert dlg.highlighted() >= 4                   # 3 legs + marker (+ candidate / preview)

    dlg.lst_legs.setCurrentRow(legs_on_b.index(('82', 'start')))
    assert dlg.btn_detach.isEnabled()
    monkeypatch.setattr(dlg, '_confirm', lambda *_a: False)
    dlg.btn_detach.click()
    assert om.segment_data['82']['Start_Point'] == _txt(B)   # "No" changes nothing

    monkeypatch.setattr(dlg, '_confirm', lambda *_a: True)
    dlg.btn_detach.click()
    assert om.segment_data['82']['Start_Point'] != _txt(B)
    assert dlg.lst_wp.count() == n_wps + 1
    assert dlg.current_wp() == b
    assert dlg.lst_legs.count() == 2

    dlg.close()
    assert not QgsProject.instance().mapLayersByName(dlg.LAYER_NAME)
    assert dlg.highlighted() == 0
    assert getattr(om, '_waypoint_editor_dlg', None) is None


def test_lone_leg_end_cannot_be_detached_and_pick_selects(hqi, legs):
    from omrat_utils.waypoint_editor_dialog import open_waypoint_editor
    om = hqi.omrat
    dlg = open_waypoint_editor(om)
    a = _wp_at(om, A)
    assert dlg.pick_at((A[0] + 0.00001, A[1])) == a
    assert dlg.current_wp() == a
    assert dlg.lst_legs.count() == 1
    assert not dlg.btn_detach.isEnabled()
    assert dlg.pick_at((10.0, 50.0)) is None


def _ask(monkeypatch, answer):
    """Answer the "Legs now cross -- split now?" prompt; returns the texts asked."""
    from omrat_utils import route_validation_ui
    asked = []
    monkeypatch.setattr(route_validation_ui, '_ask_split_now',
                        lambda _om, text: asked.append(text) or answer)
    return asked


def _drag_end(hqi, leg_id, old_end, new_end):
    """What on_geometry_changed does for an end-vertex drag of ``leg_id``."""
    seg = hqi.omrat.segment_data[leg_id]
    old_start = tuple(float(v) for v in seg['Start_Point'].split())
    seg['End_Point'] = _txt(new_end)
    hqi._apply_vertex_move(int(leg_id), old_start, old_end, old_start, new_end)


def test_drag_that_creates_a_crossing_offers_the_split(hqi, legs, monkeypatch):
    """Nelly's case (ESCOW, 2026-10-16): a leg end dragged across another
    leg used to stay an unsplit X-crossing -- zero crossing collisions."""
    om = hqi.omrat
    asked = _ask(monkeypatch, True)
    _drag_end(hqi, '84', E, (13.8, 55.0))           # C -> (13.8, 55.0) crosses 81 (A -> B)
    assert len(asked) == 1 and 'LEG_1_81 x LEG_1_84' in asked[0]
    from geometries.route_validation import find_leg_intersections
    assert find_leg_intersections(om.segment_data) == []
    assert len(om.segment_data) == 6                # 81 and 84 split in two
    names = sorted(s['Leg_name'] for s in om.segment_data.values())
    assert names == ['LEG_1_81_a', 'LEG_1_81_b', 'LEG_1_82', 'LEG_1_83', 'LEG_1_84_a', 'LEG_1_84_b']


def test_drag_crossing_not_now_leaves_the_legs(hqi, legs, monkeypatch):
    om = hqi.omrat
    asked = _ask(monkeypatch, False)
    _drag_end(hqi, '84', E, (13.8, 55.0))
    assert len(asked) == 1
    assert sorted(om.segment_data) == ['81', '82', '83', '84']


def test_drag_without_new_crossing_does_not_ask(hqi, legs, monkeypatch):
    asked = _ask(monkeypatch, True)
    _drag_end(hqi, '84', E, (14.5, 55.25))
    assert asked == []


def test_connect_that_creates_a_crossing_offers_the_split(hqi, legs, monkeypatch):
    from qgis.PyQt.QtCore import Qt
    from omrat_utils.waypoint_editor_dialog import open_waypoint_editor
    om = hqi.omrat
    asked = _ask(monkeypatch, True)
    dlg = open_waypoint_editor(om)
    d = _wp_at(om, D)
    assert dlg.select_waypoint(d, zoom=False)
    cands = [tuple(dlg.cmb_cand.itemData(i, Qt.ItemDataRole.UserRole)) for i in range(dlg.cmb_cand.count())]
    dlg.cmb_cand.setCurrentIndex(cands.index(('84', 'start')))   # D -> E crosses 82 (B -> C)
    monkeypatch.setattr(dlg, '_confirm', lambda *_a: True)
    dlg.btn_attach.click()
    assert len(asked) == 1 and 'LEG_1_82 x LEG_1_84' in asked[0]
    from geometries.route_validation import find_leg_intersections
    assert find_leg_intersections(om.segment_data) == []
    assert len(om.segment_data) == 6


def test_run_model_offers_to_split_existing_crossings(hqi, legs, monkeypatch):
    from omrat_utils import route_validation_ui
    from omrat_utils.route_validation_ui import check_crossings_before_run
    om = hqi.omrat
    om.segment_data['84']['End_Point'] = _txt((13.8, 55.0))         # an X-crossing in the file
    answers = iter(['cancel', 'run', 'split'])
    monkeypatch.setattr(route_validation_ui, '_ask_run_choice', lambda _om, _text: next(answers))
    assert check_crossings_before_run(om, ('drifting',)) == 'run'    # no collisions -> no question
    assert check_crossings_before_run(om, ('collision',)) == 'cancel'
    assert check_crossings_before_run(om, ('collision',)) == 'run'
    assert len(om.segment_data) == 4
    assert check_crossings_before_run(om, ('collision',)) == 'split'
    assert len(om.segment_data) == 6
    assert check_crossings_before_run(om, ('collision',)) == 'run'   # nothing left to ask


def test_settings_menu_has_the_editor(hqi):
    from qgis.PyQt.QtWidgets import QMenuBar
    om = hqi.omrat
    texts = [
        act.text()
        for bar in om.main_widget.findChildren(QMenuBar)
        for menu_act in bar.actions() if menu_act.menu() is not None
        for act in menu_act.menu().actions()
    ]
    assert "Waypoint editor..." in texts
    assert texts.index("Waypoint editor...") + 1 == texts.index("Junction transition matrix...")
