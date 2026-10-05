"""Difference layers between two runs' result GeoPackages (Compare tab).

The per-run GeoPackage holds one feature per obstacle edge (drifting /
powered layers), per leg (``collision_lines``) or per waypoint
(``collision_points``), and two runs of the same project produce the same
keys.  Joining A and B on those keys gives, per feature, the absolute
change ``delta = B - A`` in events/year, which is what a "where did the
risk move?" map has to show.

**Style on ``delta``, not on ``pct_change``.**  Relative change is
dominated by features whose probability is negligible in both runs (a
wind-rose direction losing weight makes thousands of far-away edges show
``-40 %`` while contributing nothing).  ``delta_share_pct`` -- the
feature's share of the total change of that layer -- is the honest
relative number.

``diff_rows`` is pure and unit-tested; ``build_diff_layers`` /
``diff_layer_from_layers`` do the QGIS work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    'DIFF_SPECS', 'DiffSpec', 'DiffSummary', 'diff_rows',
    'diff_layer_from_layers', 'build_diff_layers', 'add_diff_group',
]


@dataclass(frozen=True)
class DiffSpec:
    layer_name: str
    key_fields: tuple[str, ...]
    value_field: str            # the field the map is styled on
    extra_fields: tuple[str, ...] = ()   # also diffed, carried as <name>_a/_b/_delta
    geometry: str = 'LineString'


# One entry per layer ``omrat_utils.run_persistence._LAYER_NAMES`` writes.
DIFF_SPECS: tuple[DiffSpec, ...] = (
    DiffSpec('drifting_allision', ('obstacle_id', 'segment_idx'), 'total_edge_probability',
             ('object_probability',)),
    DiffSpec('drifting_grounding', ('obstacle_id', 'segment_idx'), 'total_edge_probability',
             ('object_probability',)),
    DiffSpec('powered_allision', ('obstacle_id', 'segment_idx'), 'total_edge_probability',
             ('object_probability',)),
    DiffSpec('powered_grounding', ('obstacle_id', 'segment_idx'), 'total_edge_probability',
             ('object_probability',)),
    DiffSpec('collision_lines', ('leg_id',), 'combined', ('head_on', 'overtaking')),
    DiffSpec('collision_points', ('waypoint',), 'combined', ('crossing', 'merging', 'bend'), 'Point'),
)


@dataclass
class DiffSummary:
    layer_name: str
    n_features: int = 0
    n_changed: int = 0
    sum_a: float = 0.0
    sum_b: float = 0.0
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_delta(self) -> float:
        return self.sum_b - self.sum_a


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out == out else 0.0  # NaN -> 0


def _key(row: Mapping[str, Any], key_fields: tuple[str, ...]) -> tuple:
    out = []
    for k in key_fields:
        v = row.get(k)
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        out.append(str(v) if not isinstance(v, int) else v)
    return tuple(out)


def diff_rows(
    rows_a: Iterable[Mapping[str, Any]],
    rows_b: Iterable[Mapping[str, Any]],
    spec: DiffSpec,
    *,
    changed_eps: float = 1e-15,
) -> DiffSummary:
    """Join two runs' rows on ``spec.key_fields`` and diff ``spec.value_field``.

    Every input row is a mapping with the key fields, the value fields and
    optionally ``'geometry'`` (WKT or a QgsGeometry; passed through from
    whichever side has it).  Output rows carry the key fields,
    ``p_a`` / ``p_b`` / ``delta`` / ``ratio`` / ``pct_change`` /
    ``delta_share_pct`` / ``rank`` (1 = largest absolute change) and, per
    extra field, ``<name>_a`` / ``<name>_b`` / ``<name>_delta``.
    """
    a_by = {_key(r, spec.key_fields): r for r in rows_a}
    b_by = {_key(r, spec.key_fields): r for r in rows_b}
    summary = DiffSummary(layer_name=spec.layer_name)
    out: list[dict[str, Any]] = []
    for key in a_by.keys() | b_by.keys():
        ra = a_by.get(key) or {}
        rb = b_by.get(key) or {}
        pa = _f(ra.get(spec.value_field))
        pb = _f(rb.get(spec.value_field))
        delta = pb - pa
        row: dict[str, Any] = {k: v for k, v in zip(spec.key_fields, key)}
        row.update({
            'p_a': pa, 'p_b': pb, 'delta': delta,
            'ratio': (pb / pa) if pa > 0 else None,
            'pct_change': ((pb - pa) / pa * 100.0) if pa > 0 else None,
            'in_a': bool(ra), 'in_b': bool(rb),
        })
        for name in spec.extra_fields:
            xa, xb = _f(ra.get(name)), _f(rb.get(name))
            row[f'{name}_a'] = xa
            row[f'{name}_b'] = xb
            row[f'{name}_delta'] = xb - xa
        geom = ra.get('geometry') if ra.get('geometry') is not None else rb.get('geometry')
        if geom is not None:
            row['geometry'] = geom
        out.append(row)
        summary.sum_a += pa
        summary.sum_b += pb
        if abs(delta) > changed_eps:
            summary.n_changed += 1
    total = summary.total_delta
    out.sort(key=lambda r: -abs(r['delta']))
    for i, row in enumerate(out, start=1):
        row['rank'] = i
        row['delta_share_pct'] = (row['delta'] / total * 100.0) if total != 0 else None
    summary.rows = out
    summary.n_features = len(out)
    return summary


# ---------------------------------------------------------------------------
# QGIS side
# ---------------------------------------------------------------------------

_OUT_FIELDS = (
    ('p_a', 'double'), ('p_b', 'double'), ('delta', 'double'), ('ratio', 'double'),
    ('pct_change', 'double'), ('delta_share_pct', 'double'), ('rank', 'integer'),
    ('in_a', 'integer'), ('in_b', 'integer'),
)


def _layer_rows(layer, spec: DiffSpec) -> list[dict[str, Any]]:
    names = [f.name() for f in layer.fields()]
    wanted = set(spec.key_fields) | {spec.value_field} | set(spec.extra_fields)
    rows: list[dict[str, Any]] = []
    for feat in layer.getFeatures():
        row = {n: feat[n] for n in names if n in wanted}
        # QVariant NULL -> None
        for k, v in list(row.items()):
            if v is not None and not isinstance(v, (int, float, str)):
                try:
                    row[k] = None if v.isNull() else v
                except AttributeError:
                    row[k] = str(v)
        geom = feat.geometry()
        if geom is not None and not geom.isEmpty():
            row['geometry'] = geom
        rows.append(row)
    return rows


def _pretty(layer_name: str) -> str:
    return layer_name.replace('_', ' ').title()


def diff_layer_from_layers(layer_a, layer_b, spec: DiffSpec, title: str):
    """Memory layer with the diff of two loaded result layers (or ``None``).

    Returns ``(layer, summary)``; ``layer`` is ``None`` when both inputs
    are empty.
    """
    from qgis.core import (
        QgsFeature, QgsField, QgsGeometry, QgsVectorLayer,
    )
    from qgis.PyQt.QtCore import QMetaType

    rows_a = _layer_rows(layer_a, spec) if layer_a is not None else []
    rows_b = _layer_rows(layer_b, spec) if layer_b is not None else []
    summary = diff_rows(rows_a, rows_b, spec)
    if not summary.rows:
        return None, summary

    crs = None
    for src in (layer_a, layer_b):
        if src is not None and src.crs().isValid():
            crs = src.crs().authid()
            break
    crs = crs or 'EPSG:4326'
    layer = QgsVectorLayer(f'{spec.geometry}?crs={crs}', title, 'memory')
    provider = layer.dataProvider()

    def _qfield(name: str, kind: str) -> QgsField:
        if kind == 'double':
            return QgsField(name, QMetaType.Type.Double)
        if kind == 'integer':
            return QgsField(name, QMetaType.Type.Int)
        return QgsField(name, QMetaType.Type.QString)

    fields = [_qfield(k, 'string') for k in spec.key_fields]
    fields += [_qfield(n, t) for n, t in _OUT_FIELDS]
    for name in spec.extra_fields:
        fields += [_qfield(f'{name}_a', 'double'), _qfield(f'{name}_b', 'double'),
                   _qfield(f'{name}_delta', 'double')]
    provider.addAttributes(fields)
    layer.updateFields()
    names = [f.name() for f in layer.fields()]

    feats = []
    for row in summary.rows:
        feat = QgsFeature(layer.fields())
        values = []
        for n in names:
            v = row.get(n)
            if isinstance(v, bool):
                v = int(v)
            elif n in spec.key_fields and v is not None:
                v = str(v)
            values.append(v)
        feat.setAttributes(values)
        geom = row.get('geometry')
        if geom is not None:
            if isinstance(geom, str):
                geom = QgsGeometry.fromWkt(geom)
            feat.setGeometry(geom)
        feats.append(feat)
    provider.addFeatures(feats)
    layer.updateExtents()
    apply_diverging_style(layer, 'delta')
    return layer, summary


def apply_diverging_style(layer, attribute: str = 'delta', num_classes: int = 4) -> None:
    """Blue (decrease) / grey (no change) / red (increase), symmetric about 0.

    Class edges are placed at fractions of the largest absolute change so
    the legend reads as a share of the biggest mover, in events/year.
    """
    try:
        from qgis.core import (
            Qgis, QgsGraduatedSymbolRenderer, QgsLineSymbol, QgsMarkerSymbol,
            QgsRendererRange,
        )
    except Exception:  # nosec B110 B112
        return
    values = [abs(f[attribute]) for f in layer.getFeatures()
              if f[attribute] is not None and f[attribute] == f[attribute]]
    vmax = max(values) if values else 0.0
    if vmax <= 0:
        return
    is_point = layer.geometryType() == Qgis.GeometryType.Point

    def _symbol(color: str, width: float):
        if is_point:
            return QgsMarkerSymbol.createSimple({'color': color, 'size': str(1.5 + width * 2), 'outline_style': 'no'})
        return QgsLineSymbol.createSimple({'color': color, 'width': str(width)})

    reds = ('#fdd0a2', '#fd8d3c', '#e6550d', '#a63603')
    blues = ('#c6dbef', '#6baed6', '#3182bd', '#08519c')
    edges = [vmax * frac for frac in (0.0, 0.01, 0.1, 0.5, 1.0)]
    eps = vmax * 1e-6
    ranges = []
    for i in range(num_classes - 1, -1, -1):
        lo, hi = edges[i], edges[i + 1]
        ranges.append(QgsRendererRange(
            -hi * 1.0001, -lo if i else -eps, _symbol(blues[i], 0.4 + 0.3 * i),
            f'{-hi:.2e} to {-lo:.2e}' if i else f'decrease down to {-hi:.2e}'))
    ranges.append(QgsRendererRange(-eps, eps, _symbol('#bdbdbd', 0.25), 'no change'))
    for i in range(num_classes):
        lo, hi = edges[i], edges[i + 1]
        ranges.append(QgsRendererRange(
            eps if i == 0 else lo, hi * 1.0001, _symbol(reds[i], 0.4 + 0.3 * i),
            f'{lo:.2e} to {hi:.2e}' if i else f'increase up to {hi:.2e}'))
    renderer = QgsGraduatedSymbolRenderer(attribute, ranges)
    layer.setRenderer(renderer)
    layer.triggerRepaint()


def _open_gpkg_layer(gpkg: Path, layer_name: str):
    from qgis.core import QgsVectorLayer
    layer = QgsVectorLayer(f'{gpkg}|layername={layer_name}', layer_name, 'ogr')
    if not layer.isValid():
        return None
    return layer


def build_diff_layers(
    gpkg_a: str | Path, gpkg_b: str | Path, label: str = 'B - A',
) -> tuple[list[Any], list[DiffSummary]]:
    """One diff memory layer per result layer present in either GeoPackage."""
    gpkg_a, gpkg_b = Path(gpkg_a), Path(gpkg_b)
    layers: list[Any] = []
    summaries: list[DiffSummary] = []
    for spec in DIFF_SPECS:
        la = _open_gpkg_layer(gpkg_a, spec.layer_name) if gpkg_a.is_file() else None
        lb = _open_gpkg_layer(gpkg_b, spec.layer_name) if gpkg_b.is_file() else None
        if la is None and lb is None:
            continue
        layer, summary = diff_layer_from_layers(la, lb, spec, f'Diff {_pretty(spec.layer_name)} ({label})')
        summaries.append(summary)
        if layer is not None:
            layers.append(layer)
    return layers, summaries


def add_diff_group(layers: Iterable[Any], group_name: str) -> dict[str, Any]:
    """Put ``layers`` in a new layer-tree group at the top of the project.

    Same return shape as ``snapshot_layers.add_snapshot_to_project`` so
    ``CompareMixin._remember_compare_group`` / ``clear_model`` handle it.
    """
    from qgis.core import QgsProject
    project = QgsProject.instance()
    group = project.layerTreeRoot().insertGroup(0, group_name)
    added = []
    for layer in layers:
        project.addMapLayer(layer, False)
        group.insertLayer(0, layer)
        added.append(layer)
    return {'group': group, 'model': [], 'results': added}


def summary_lines(summaries: Iterable[DiffSummary]) -> list[str]:
    lines = []
    for s in summaries:
        if s.n_features == 0:
            continue
        pct = f' ({s.total_delta / s.sum_a * 100:+.1f} %)' if s.sum_a > 0 else ''
        lines.append(
            f'{_pretty(s.layer_name)}: {s.n_changed} of {s.n_features} features changed, '
            f'total {s.sum_a:.3e} -> {s.sum_b:.3e}{pct}')
    return lines
