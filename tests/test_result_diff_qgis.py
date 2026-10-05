"""QGIS-fixture tests for the Compare-tab difference layers.

Needs the conftest (``QgsApplication``): run *without* ``--noconftest``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from qgis.core import (
    QgsCoordinateTransformContext, QgsFeature, QgsField, QgsGeometry,
    QgsGraduatedSymbolRenderer, QgsProject, QgsVectorFileWriter, QgsVectorLayer,
)
from qgis.PyQt.QtCore import QMetaType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.result_diff import (  # noqa: E402
    DIFF_SPECS, build_diff_layers, diff_layer_from_layers,
)

EDGE = next(s for s in DIFF_SPECS if s.layer_name == 'drifting_grounding')
POINTS = next(s for s in DIFF_SPECS if s.layer_name == 'collision_points')


def _edge_layer(name: str, values: dict[tuple[str, int], float]) -> QgsVectorLayer:
    layer = QgsVectorLayer('LineString?crs=EPSG:4326', name, 'memory')
    prov = layer.dataProvider()
    prov.addAttributes([
        QgsField('obstacle_id', QMetaType.Type.QString), QgsField('segment_idx', QMetaType.Type.Int),
        QgsField('total_edge_probability', QMetaType.Type.Double),
        QgsField('object_probability', QMetaType.Type.Double),
    ])
    layer.updateFields()
    feats = []
    for (obs, idx), p in values.items():
        f = QgsFeature(layer.fields())
        f.setAttributes([obs, idx, p, p * 2])
        f.setGeometry(QgsGeometry.fromWkt(f'LINESTRING({idx} 0, {idx} 1)'))
        feats.append(f)
    prov.addFeatures(feats)
    layer.updateExtents()
    return layer


def _write_gpkg(path: Path, layers: list[QgsVectorLayer]) -> None:
    for i, layer in enumerate(layers):
        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = 'GPKG'
        opts.layerName = layer.name()
        if i:
            opts.actionOnExistingFile = QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
        res = QgsVectorFileWriter.writeAsVectorFormatV3(layer, str(path), QgsCoordinateTransformContext(), opts)
        assert res[0] == QgsVectorFileWriter.WriterError.NoError, res


def test_diff_layer_from_memory_layers():
    a = _edge_layer('drifting_grounding', {('d3', 1): 1e-6, ('d3', 2): 2e-6, ('d6', 1): 5e-7})
    b = _edge_layer('drifting_grounding', {('d3', 1): 2e-6, ('d3', 2): 2e-6, ('d6', 1): 0.0})
    layer, summary = diff_layer_from_layers(a, b, EDGE, 'Diff test')
    assert layer is not None and layer.isValid()
    assert layer.featureCount() == 3
    assert summary.n_changed == 2
    names = [f.name() for f in layer.fields()]
    for n in ('obstacle_id', 'segment_idx', 'p_a', 'p_b', 'delta', 'ratio', 'pct_change',
              'delta_share_pct', 'rank', 'object_probability_delta'):
        assert n in names
    by = {(f['obstacle_id'], int(f['segment_idx'])): f for f in layer.getFeatures()}
    assert by[('d3', 1)]['delta'] == pytest.approx(1e-6)
    assert by[('d3', 1)]['rank'] == 1
    assert by[('d6', 1)]['delta'] == pytest.approx(-5e-7)
    assert by[('d3', 2)]['delta'] == 0.0
    # Geometry carried over from A.
    assert by[('d3', 1)].geometry().asWkt().startswith('LineString (1 0, 1 1)')
    # Styled on delta with a diverging ramp: negative, zero and positive classes.
    renderer = layer.renderer()
    assert isinstance(renderer, QgsGraduatedSymbolRenderer)
    assert renderer.classAttribute() == 'delta'
    lows = [r.lowerValue() for r in renderer.ranges()]
    assert min(lows) < 0 < max(lows)
    assert any(r.label() == 'no change' for r in renderer.ranges())


def test_build_diff_layers_from_gpkgs(tmp_path):
    pa = tmp_path / 'a.gpkg'
    pb = tmp_path / 'b.gpkg'
    _write_gpkg(pa, [_edge_layer('drifting_grounding', {('d3', 1): 1e-6})])
    _write_gpkg(pb, [_edge_layer('drifting_grounding', {('d3', 1): 3e-6})])
    layers, summaries = build_diff_layers(pa, pb, label='B - A')
    assert [lyr.name() for lyr in layers] == ['Diff Drifting Grounding (B - A)']
    assert summaries[0].total_delta == pytest.approx(2e-6)
    f = next(layers[0].getFeatures())
    assert f['delta'] == pytest.approx(2e-6) and f['ratio'] == pytest.approx(3.0)


def test_compare_tab_button_adds_group_and_clear_model_removes_it(omrat, tmp_path):
    w = omrat.main_widget
    assert hasattr(w, 'pbCompareDiff')
    pa = tmp_path / 'runA.gpkg'
    pb = tmp_path / 'runB.gpkg'
    _write_gpkg(pa, [_edge_layer('drifting_grounding', {('d3', 1): 1e-6, ('d3', 2): 1e-6})])
    _write_gpkg(pb, [_edge_layer('drifting_grounding', {('d3', 1): 5e-6, ('d3', 2): 1e-6})])
    (tmp_path / 'runA.omrat').write_text('{}', encoding='utf-8')
    (tmp_path / 'runB.omrat').write_text('{}', encoding='utf-8')
    w.LECompareA.setText(str(tmp_path / 'runA.omrat'))
    w.LECompareB.setText(str(tmp_path / 'runB.omrat'))
    w.pbCompareDiff.click()
    root = QgsProject.instance().layerTreeRoot()
    groups = [g for g in root.children() if g.name().startswith('Compare diff:')]
    assert len(groups) == 1
    assert groups[0].name() == 'Compare diff: runB - runA'
    layer_names = [c.layer().name() for c in groups[0].children()]
    assert layer_names == ['Diff Drifting Grounding (B - A)']
    assert 'Difference layers (B - A) added: 1' in w.lblCompareSummary.text()
    assert '1 of 2 features changed' in w.lblCompareSummary.text()
    omrat.clear_model()
    assert not [g for g in root.children() if g.name().startswith('Compare diff:')]
