"""Layer-style persistence (QGIS fixture): restyle in QGIS -> Save -> Load."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _dist_keys() -> dict:
    """Distribution parameters the .omrat schema requires per leg."""
    d = {}
    for j in (1, 2):
        for i in (1, 2, 3):
            d[f'mean{j}_{i}'] = 0.0
            d[f'std{j}_{i}'] = 0.0
            d[f'weight{j}_{i}'] = 100.0 if i == 1 else 0.0
        d[f'u_min{j}'] = 0.0
        d[f'u_max{j}'] = 0.0
        d[f'u_p{j}'] = 0
        d[f'ai{j}'] = 180
    return d


SEG = {
    '77': {
        'Start_Point': '14.000000 55.000000', 'End_Point': '14.200000 55.000000',
        'Width': 5000, 'Route_Id': 1, 'Leg_name': 'LEG_1_77', 'Segment_Id': '77',
        'Dirs': ['East going', 'West going'], 'line_length': 12000.0, 'Tangent_Pos': 0.5,
        **_dist_keys(),
    },
    '78': {
        'Start_Point': '14.200000 55.000000', 'End_Point': '14.400000 55.000000',
        'Width': 5000, 'Route_Id': 1, 'Leg_name': 'LEG_1_78', 'Segment_Id': '78',
        'Dirs': ['East going', 'West going'], 'line_length': 12000.0, 'Tangent_Pos': 0.5,
        **_dist_keys(),
    },
}


def _leg_color(layer) -> str:
    return layer.renderer().symbol().color().name()


def _set_leg_color(layer, hex_color: str, width: float | None = None) -> None:
    from qgis.PyQt.QtGui import QColor
    sym = layer.renderer().symbol()
    sym.setColor(QColor(hex_color))
    if width is not None:
        sym.setWidth(width)
    layer.triggerRepaint()


@pytest.fixture
def legs(omrat):
    import copy
    omrat.segment_data = copy.deepcopy(SEG)
    omrat.load_lines({'segment_data': omrat.segment_data})
    from omrat_utils.layer_styles import leg_layers, tangent_layer
    layers = leg_layers(omrat)
    assert len(layers) == 2
    assert tangent_layer(omrat) is not None
    return layers


class TestExportApply:
    def test_round_trip_symbol_color(self, omrat, legs):
        from omrat_utils.layer_styles import export_style, apply_style
        a, b = legs
        _set_leg_color(a, '#ff0000', 2.5)
        qml = export_style(a)
        assert qml and '<qgis' in qml
        assert apply_style(b, qml) is True
        assert _leg_color(b) == '#ff0000'
        assert b.renderer().symbol().width() == pytest.approx(2.5)

    def test_custom_properties_are_not_exported(self, omrat, legs):
        from omrat_utils.layer_styles import export_style
        a, _b = legs
        a.setCustomProperty('segment_id', 77)
        qml = export_style(a)
        assert 'segment_id' not in qml

    def test_apply_rejects_garbage(self, omrat, legs):
        from omrat_utils.layer_styles import apply_style
        a, _b = legs
        assert apply_style(a, 'not xml') is False
        assert apply_style(a, '') is False
        assert apply_style(None, '<qgis/>') is False


class TestCollectApply:
    def test_collect_has_legs_and_tangent(self, omrat, legs):
        from omrat_utils.layer_styles import collect_styles
        styles = collect_styles(omrat)
        assert set(styles) >= {'legs', 'tangent'}
        assert 'depths' not in styles and 'structures' not in styles
        json.dumps(styles)  # serialisable

    def test_apply_styles_hits_every_leg_and_remembers(self, omrat, legs):
        from omrat_utils.layer_styles import collect_styles, apply_styles
        a, b = legs
        _set_leg_color(a, '#00ff00')
        styles = collect_styles(omrat)
        _set_leg_color(a, '#0000ff')
        _set_leg_color(b, '#0000ff')
        applied = apply_styles(omrat, styles)
        assert applied['legs'] == 2
        assert _leg_color(a) == '#00ff00' and _leg_color(b) == '#00ff00'
        assert omrat.layer_styles == styles

    def test_collect_keeps_remembered_style_for_absent_layer_type(self, omrat, legs):
        from omrat_utils.layer_styles import collect_styles
        omrat.layer_styles = {'structures': '<qgis>kept</qgis>'}
        styles = collect_styles(omrat)
        assert styles['structures'] == '<qgis>kept</qgis>'

    def test_new_leg_after_load_gets_stored_style(self, omrat, legs):
        from omrat_utils.layer_styles import collect_styles, apply_styles, leg_layers
        a, _b = legs
        _set_leg_color(a, '#ff00ff')
        apply_styles(omrat, collect_styles(omrat))
        omrat.segment_data['79'] = dict(SEG['78'], Segment_Id='79', Leg_name='LEG_1_79',
                                        Start_Point='14.400000 55.000000', End_Point='14.600000 55.000000')
        # A leg created through the interactive path applies the stored style.
        from qgis.core import QgsVectorLayer
        vl = QgsVectorLayer("LineString?crs=EPSG:4326", "LEG_1_79", "memory")
        omrat.qgis_geoms.style_layer(vl)
        from omrat_utils.layer_styles import apply_stored_style
        assert apply_stored_style(omrat, 'legs', vl) is True
        assert _leg_color(vl) == '#ff00ff'
        assert len(leg_layers(omrat)) == 2  # helper layer was never registered


class TestSaveLoadRoundTrip:
    def test_style_survives_save_and_load(self, omrat, legs, monkeypatch, tmp_path):
        from omrat_utils.layer_styles import leg_layers, tangent_layer
        from omrat_utils.storage import Storage
        a, _b = legs
        _set_leg_color(a, '#ff8800', 3.0)
        _set_leg_color(tangent_layer(omrat), '#123456')
        target = tmp_path / 'styled.omrat'
        assert Storage(omrat).store_all(str(target)) == str(target)
        saved = json.loads(target.read_text())
        assert set(saved['layer_styles']) >= {'legs', 'tangent'}

        omrat.clear_model()
        assert leg_layers(omrat) == []
        Storage(omrat).load_from_path(str(target))
        new_legs = leg_layers(omrat)
        assert len(new_legs) == 2
        for lyr in new_legs:
            assert _leg_color(lyr) == '#ff8800'
            assert lyr.renderer().symbol().width() == pytest.approx(3.0)
        assert _leg_color(tangent_layer(omrat)) == '#123456'
        assert omrat.layer_styles == saved['layer_styles']

    def test_restyle_counts_as_unsaved_change(self, omrat, legs):
        a, _b = legs
        omrat.mark_project_saved()
        assert omrat.has_unsaved_changes() is False
        _set_leg_color(a, '#abcdef')
        assert omrat.has_unsaved_changes() is True

    def test_project_without_block_uses_defaults(self, omrat, legs, monkeypatch, tmp_path):
        from omrat_utils.storage import Storage
        from omrat_utils.layer_styles import leg_layers
        target = tmp_path / 'plain.omrat'
        Storage(omrat).store_all(str(target))
        data = json.loads(target.read_text())
        data.pop('layer_styles', None)
        target.write_text(json.dumps(data))
        omrat.clear_model()
        Storage(omrat).load_from_path(str(target))
        assert len(leg_layers(omrat)) == 2
        assert _leg_color(leg_layers(omrat)[0]) == '#0000ff'   # style_layer default: blue
