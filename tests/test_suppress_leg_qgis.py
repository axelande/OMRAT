"""QGIS tests for **Suppress leg...**: flag, dashed map layer, restore,
the modeless dialog and the project leg style.  Needs the QGIS conftest."""
from __future__ import annotations

import copy

import pytest


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


def _pen_styles(layer):
    return [sl.penStyle() for sl in layer.renderer().symbol().symbolLayers()]


@pytest.fixture
def hqi(omrat):
    return omrat.qgis_geoms


@pytest.fixture
def legs(hqi):
    """Leg 91 (north) with traffic, leg 92 continuing it; both on the map."""
    from qgis.core import QgsProject
    om = hqi.omrat
    om.testing = True
    base = {'Width': 2000, 'Route_Id': 1, 'Tangent_Pos': 0.5, 'Dirs': ['North going', 'South going']}
    om.segment_data['91'] = dict(copy.deepcopy(base), Segment_Id='91', Leg_name='LEG_1_91',
                                 Start_Point='14.000000 55.000000', End_Point='14.000000 55.200000')
    om.segment_data['92'] = dict(copy.deepcopy(base), Segment_Id='92', Leg_name='LEG_1_92',
                                 Start_Point='14.000000 55.200000', End_Point='14.000000 55.400000')
    om.traffic.create_empty_dict('91', ['North going', 'South going'])
    om.traffic.create_empty_dict('92', ['North going', 'South going'])
    om.traffic_data['91']['North going']['Frequency (ships/year)'][0][0] = 50
    layers = [_make_leg_layer(hqi, k, om.segment_data[k]['Start_Point'], om.segment_data[k]['End_Point'])
              for k in ('91', '92')]
    yield layers
    for lyr in layers:
        hqi.vector_layers.remove(lyr)
        QgsProject.instance().removeMapLayer(lyr.id())
    for k in ('91', '92'):
        om.segment_data.pop(k, None)
        om.traffic_data.pop(k, None)
    dlg = getattr(om, '_suppress_leg_dlg', None)
    if dlg is not None:
        dlg.close()
        om._suppress_leg_dlg = None


def test_suppress_dashes_the_leg_and_restore_keeps_redirect(hqi, legs):
    from qgis.PyQt.QtCore import Qt
    from omrat_utils.suppress_leg_dialog import apply_suppression
    redirect = [{'from_dir': 0, 'leg': '92', 'dir': 0, 'share': 100}]
    apply_suppression(hqi.omrat, '91', True, redirect)
    seg = hqi.omrat.segment_data['91']
    assert seg['suppressed'] is True
    assert seg['traffic_redirect'][0]['leg'] == '92'
    assert _pen_styles(legs[0]) == [Qt.PenStyle.DashLine]
    assert _pen_styles(legs[1]) == [Qt.PenStyle.SolidLine]

    apply_suppression(hqi.omrat, '91', False)
    assert seg['suppressed'] is False
    assert seg['traffic_redirect'][0]['leg'] == '92'   # kept for a later re-suppress
    assert _pen_styles(legs[0]) == [Qt.PenStyle.SolidLine]


def test_button_wired(hqi):
    btn = getattr(hqi.omrat.main_widget, 'pbSuppressLeg', None)
    assert btn is not None
    assert btn.receivers(btn.clicked) > 0


def test_dialog_is_modeless_and_prefills_direction(hqi, legs):
    from omrat_utils import suppress_leg_dialog
    suppress_leg_dialog.run(hqi.omrat)
    dlg = hqi.omrat._suppress_leg_dlg
    assert dlg.isVisible() and not dlg.isModal()
    dlg.cb_src.setCurrentIndex(dlg.cb_src.findData('91'))
    row = dlg.add_row()
    dlg.table.cellWidget(row, 1).setCurrentIndex(dlg.table.cellWidget(row, 1).findData('92'))
    dlg.table.cellWidget(row, 0).setCurrentIndex(1)            # South going on 91 ...
    assert dlg.table.cellWidget(row, 2).currentData() == 1      # ... lands on South going of 92
    dlg.table.cellWidget(row, 3).setValue(40)
    assert dlg.entries() == [{'from_dir': 1, 'leg': '92', 'dir': 1, 'share': 40.0}]
    # A second click raises the same dialog instead of opening another.
    suppress_leg_dialog.run(hqi.omrat)
    assert hqi.omrat._suppress_leg_dlg is dlg


def test_saved_leg_style_ignores_suppressed_leg(hqi, legs):
    from qgis.PyQt.QtCore import Qt
    from qgis.core import QgsVectorLayer
    from omrat_utils.layer_styles import apply_style, collect_styles
    from omrat_utils.suppress_leg_dialog import apply_suppression
    apply_suppression(hqi.omrat, '91', True, [])
    hqi.vector_layers.sort(key=lambda lyr: lyr is not legs[0])   # suppressed layer first
    qml = collect_styles(hqi.omrat)['legs']
    probe = QgsVectorLayer("LineString?crs=EPSG:4326", "probe", "memory")
    assert apply_style(probe, qml)
    assert _pen_styles(probe) == [Qt.PenStyle.SolidLine]


def test_iwrap_export_warns_and_can_be_cancelled(hqi, legs, monkeypatch):
    from qgis.PyQt.QtWidgets import QMessageBox
    from omrat_utils.suppress_leg_dialog import apply_suppression
    om = hqi.omrat
    seen = []

    def _fake_warning(_parent, title, text, *_args):
        seen.append((title, text))
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, 'warning', staticmethod(_fake_warning))
    assert '91' in om._export_data_with_suppression()['segment_data']   # nothing suppressed: no popup
    assert seen == []
    apply_suppression(om, '91', True, [{'from_dir': 0, 'leg': '92', 'dir': 0, 'share': 100}])
    assert om._export_data_with_suppression() is None                   # cancelled
    assert len(seen) == 1 and 'LEG_1_91' in seen[0][1]

    monkeypatch.setattr(QMessageBox, 'warning', staticmethod(lambda *_a: QMessageBox.StandardButton.Ok))
    data = om._export_data_with_suppression()
    assert '91' not in data['segment_data']
    assert data['traffic_data']['92']['North going']['Frequency (ships/year)'][0][0] == 50
    assert om.segment_data['91']['suppressed'] is True                   # project untouched


def test_dialog_suppresses_route_with_lead_and_restores_it(hqi, legs, monkeypatch):
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QMessageBox
    from omrat_utils import suppress_leg_dialog
    om = hqi.omrat
    # 91 has northbound ships but no target here -> the "not moved" prompt.
    monkeypatch.setattr(QMessageBox, 'question', staticmethod(lambda *_a: QMessageBox.StandardButton.Yes))
    suppress_leg_dialog.run(om)
    dlg = om._suppress_leg_dlg
    dlg.fill_sources(select='91')
    dlg.set_member('92')
    dlg._on_suppress()
    assert om.segment_data['92']['suppressed'] is True
    assert om.segment_data['92']['suppressed_with'] == '91'
    assert _pen_styles(legs[1]) == [Qt.PenStyle.DashLine]
    assert dlg.pb_restore.text().endswith('(+1 with it)')

    dlg._on_restore()
    assert om.segment_data['91']['suppressed'] is False
    assert om.segment_data['92']['suppressed'] is False
    assert 'suppressed_with' not in om.segment_data['92']
    assert _pen_styles(legs[1]) == [Qt.PenStyle.SolidLine]


def test_traffic_links_layer_and_highlight(hqi, legs):
    from qgis.core import QgsProject
    om = hqi.omrat
    om.segment_data['92'].update({'traffic_source': '91', 'traffic_locked': True})
    hqi.show_traffic_links(True)
    layer = hqi.traffic_links_layer
    try:
        assert QgsProject.instance().mapLayer(layer.id()) is not None
        feats = list(layer.getFeatures())
        assert [(f['kind'], f['src'], f['dst'], f['label']) for f in feats] == [('copy', '91', '92', 'copy (locked)')]
        hqi.highlight_traffic_links('91')
        assert hqi.highlighted_legs() == 2          # the leg itself + its copy
        hqi.highlight_traffic_links('92')
        assert hqi.highlighted_legs() == 2
        # A new link shows up without toggling the view.
        from omrat_utils.suppress_leg_dialog import apply_suppression
        apply_suppression(om, '91', True, [])
        apply_suppression(om, '92', False)
        kinds = sorted(f['kind'] for f in layer.getFeatures())
        assert kinds == ['copy']                    # 91 suppressed without targets: no arrow
    finally:
        hqi.show_traffic_links(False)
    assert hqi.traffic_links_layer is None
    assert hqi.highlighted_legs() == 0


def test_traffic_links_button_is_checkable_and_wired(hqi):
    btn = getattr(hqi.omrat.main_widget, 'pbTrafficLinks', None)
    assert btn is not None and btn.isCheckable()
    assert btn.receivers(btn.toggled) > 0


def test_together_list_comes_before_the_targets(hqi, legs):
    from omrat_utils import suppress_leg_dialog
    suppress_leg_dialog.run(hqi.omrat)
    dlg = hqi.omrat._suppress_leg_dlg
    layout = dlg.layout()
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert order.index(dlg.grp_with) < order.index(dlg.grp_targets)
    assert dlg.lst_with.parent() is dlg.grp_with
    assert dlg.table.parent() is dlg.grp_targets


class _FakeSettings:
    def __init__(self, value=None):
        self.store = {} if value is None else {'omrat/suppress_leg_copy_tip_shown': value}

    def value(self, key, default=None):
        return self.store.get(key, default)

    def setValue(self, key, value):
        self.store[key] = value


def test_copy_tip_is_shown_once_and_can_open_copy_traffic(hqi, legs, monkeypatch):
    from qgis.PyQt.QtWidgets import QMessageBox
    from omrat_utils import copy_traffic_dialog, suppress_leg_dialog
    shown, opened = [], []

    def _exec(box):
        shown.append(box.text())
        box_buttons = [b for b in box.buttons() if b.text().startswith('Open Copy traffic')]
        box._clicked = box_buttons[0]
        return 0

    monkeypatch.setattr(QMessageBox, 'exec', _exec)
    monkeypatch.setattr(QMessageBox, 'clickedButton', lambda box: box._clicked)
    monkeypatch.setattr(copy_traffic_dialog, 'run', lambda om: opened.append(om))
    settings = _FakeSettings()
    assert suppress_leg_dialog.maybe_show_copy_tip(hqi.omrat, settings=settings) is True
    assert len(shown) == 1 and 'Copy traffic' in shown[0]
    assert opened == [hqi.omrat]
    # Second time: nothing.
    assert suppress_leg_dialog.maybe_show_copy_tip(hqi.omrat, settings=settings) is False
    assert len(shown) == 1
    # Windows QSettings returns the flag as the string "true".
    assert suppress_leg_dialog.maybe_show_copy_tip(hqi.omrat, settings=_FakeSettings('true')) is False
