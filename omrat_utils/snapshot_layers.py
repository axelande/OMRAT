"""Turn an ``.omrat`` snapshot into QGIS layers inside one layer-tree group.

Used by the Compare tab's "Add both models to QGIS" button.  For each
snapshot the group receives, bottom to top:

1. **Depth Areas** -- one polygon feature per depth entry (``id``, ``depth``);
2. **Structures** -- one polygon feature per object (``id``, ``height``);
3. **Legs** -- one line feature per leg (``segmentId``, ``routeId``, ``name``);
4. **Tangent Lines** -- the perpendicular width marker through each
   leg's midpoint (same construction as the live route editor);
5. the run's **result layers** from the per-run GeoPackage, when it
   exists next to the snapshot.

Layers are added in that order and each new layer is inserted at the
*top* of the group, so the on-canvas stacking matches a normally loaded
model: results over tangent lines over legs over structures over
depths.  This module is QGIS-only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geometries.tangent_position import (
    TANGENT_POS_KEY, normalize_tangent_pos, perpendicular_through_point,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot row normalisation (pure)
# ---------------------------------------------------------------------------

def _area_rows(items: Any, value_key: str) -> list[tuple[str, float | None, str]]:
    """``depths`` / ``objects`` entries -> ``[(id, value, wkt), ...]``.

    Accepts both the list form ``[id, value, wkt]`` and the dict form
    ``{'id': .., '<value_key>': .., 'polygon': ..}``.
    """
    rows: list[tuple[str, float | None, str]] = []
    if not isinstance(items, (list, tuple)):
        return rows
    for item in items:
        try:
            if isinstance(item, dict):
                ident = str(item.get('id', ''))
                raw = item.get(value_key, item.get(value_key + 's'))
                wkt = str(item.get('polygon', ''))
            else:
                ident, raw, wkt = str(item[0]), item[1], str(item[2])
        except Exception:  # nosec B110 B112
            continue
        try:
            value: float | None = float(raw)
        except (TypeError, ValueError):
            value = None
        if wkt:
            rows.append((ident, value, wkt))
    return rows


def _parse_xy(text: Any) -> tuple[float, float] | None:
    """``'12.5 56.1'`` or ``'Point (12.5 56.1)'`` -> ``(12.5, 56.1)``."""
    if not isinstance(text, str):
        return None
    if '(' in text:
        text = text.split('(', 1)[1].split(')', 1)[0]
    parts = text.replace(',', ' ').split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _leg_rows(segment_data: Any) -> list[dict[str, Any]]:
    """``segment_data`` -> list of
    ``{id, route, name, start, end, width, tangent_pos}``."""
    rows: list[dict[str, Any]] = []
    if not isinstance(segment_data, dict):
        return rows
    for key, seg in segment_data.items():
        if not isinstance(seg, dict):
            continue
        start = _parse_xy(seg.get('Start_Point'))
        end = _parse_xy(seg.get('End_Point'))
        if start is None or end is None:
            continue
        seg_id = str(seg.get('Segment_Id', key))
        route_id = str(seg.get('Route_Id', 1))
        try:
            width = float(seg.get('Width', 5000) or 5000)
        except (TypeError, ValueError):
            width = 5000.0
        rows.append({
            'id': seg_id,
            'route': route_id,
            'name': str(seg.get('Leg_name', f'LEG_{route_id}_{seg_id}')),
            'start': start,
            'end': end,
            'width': width,
            'tangent_pos': normalize_tangent_pos(seg.get(TANGENT_POS_KEY)),
        })
    rows.sort(key=lambda r: int(r['id']) if r['id'].isdigit() else 10 ** 9)
    return rows


def perpendicular_through_midpoint(
    start: tuple[float, float], end: tuple[float, float], half_width: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Endpoints of the segment perpendicular to ``start->end`` through
    its midpoint, extending ``half_width`` to each side (planar maths --
    feed it projected coordinates).  Thin wrapper over
    :func:`geometries.tangent_position.perpendicular_through_point`."""
    return perpendicular_through_point(start, end, half_width, 0.5)


# ---------------------------------------------------------------------------
# Layer builders (QGIS)
# ---------------------------------------------------------------------------

def _memory_layer(geom: str, name: str, fields: list[tuple[str, Any]]):
    from qgis.core import QgsField, QgsVectorLayer
    layer = QgsVectorLayer(f"{geom}?crs=EPSG:4326", name, "memory")
    pr = layer.dataProvider()
    if pr is not None and fields:
        pr.addAttributes([QgsField(n, t) for n, t in fields])
        layer.updateFields()
    return layer


def _field_types():
    from qgis.PyQt.QtCore import QMetaType
    return QMetaType.Type.QString, QMetaType.Type.Double


def build_area_layer(items: Any, value_key: str, name: str, *, fill: str, outline: str):
    """Polygon memory layer for ``depths`` (``value_key='depth'``) or
    ``objects`` (``value_key='height'``).  ``None`` when there is nothing."""
    from qgis.core import QgsFeature, QgsFillSymbol, QgsGeometry, QgsSingleSymbolRenderer
    rows = _area_rows(items, value_key)
    if not rows:
        return None
    t_str, t_dbl = _field_types()
    layer = _memory_layer('Polygon', name, [('id', t_str), (value_key, t_dbl)])
    pr = layer.dataProvider()
    feats = []
    for ident, value, wkt in rows:
        geom = QgsGeometry.fromWkt(wkt)
        if geom.isEmpty():
            continue
        feat = QgsFeature(layer.fields())
        feat.setGeometry(geom)
        feat.setAttributes([ident, value])
        feats.append(feat)
    if not feats:
        return None
    pr.addFeatures(feats)
    layer.updateExtents()
    try:
        sym = QgsFillSymbol.createSimple({
            'color': fill, 'outline_color': outline, 'outline_width': '0.3',
        })
        layer.setRenderer(QgsSingleSymbolRenderer(sym))
    except Exception:  # nosec B110 B112
        pass
    return layer


def build_leg_layer(segment_data: Any, name: str, *, color: str):
    """LineString memory layer with one feature per leg, labelled by name."""
    from qgis.core import (
        QgsFeature, QgsLineString, QgsLineSymbol, QgsPalLayerSettings, QgsPoint,
        QgsSingleSymbolRenderer, QgsVectorLayerSimpleLabeling,
    )
    rows = _leg_rows(segment_data)
    if not rows:
        return None
    t_str, _t_dbl = _field_types()
    layer = _memory_layer(
        'LineString', name, [('segmentId', t_str), ('routeId', t_str), ('name', t_str)],
    )
    feats = []
    for row in rows:
        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsLineString([QgsPoint(*row['start']), QgsPoint(*row['end'])]))
        feat.setAttributes([row['id'], row['route'], row['name']])
        feats.append(feat)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    try:
        sym = QgsLineSymbol.createSimple({'color': color, 'width': '0.8'})
        layer.setRenderer(QgsSingleSymbolRenderer(sym))
        settings = QgsPalLayerSettings()
        settings.fieldName = 'name'
        settings.enabled = True
        layer.setLabelsEnabled(True)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    except Exception:  # nosec B110 B112
        pass
    return layer


def _utm_transforms(lon: float, lat: float):
    from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
    zone = int((lon + 180) / 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    wgs84 = QgsCoordinateReferenceSystem('EPSG:4326')
    utm = QgsCoordinateReferenceSystem(f'EPSG:{epsg}')
    ctx = QgsProject.instance()
    return QgsCoordinateTransform(wgs84, utm, ctx), QgsCoordinateTransform(utm, wgs84, ctx)


def build_tangent_layer(segment_data: Any, name: str, *, color: str):
    """LineString memory layer with the width marker of every leg.

    The perpendicular is built in the leg's UTM zone (metres) and
    transformed back to WGS84, matching ``HandleQGISIface.create_offset_lines``.
    """
    from qgis.core import (
        QgsFeature, QgsLineString, QgsLineSymbol, QgsPoint, QgsPointXY,
        QgsSingleSymbolRenderer,
    )
    rows = _leg_rows(segment_data)
    if not rows:
        return None
    t_str, _t_dbl = _field_types()
    layer = _memory_layer('LineString', name, [('type', t_str)])
    feats = []
    for row in rows:
        try:
            to_utm, to_wgs = _utm_transforms(
                (row['start'][0] + row['end'][0]) / 2.0, row['start'][1],
            )
            s_utm = to_utm.transform(QgsPointXY(*row['start']))
            e_utm = to_utm.transform(QgsPointXY(*row['end']))
            ends = perpendicular_through_point(
                (s_utm.x(), s_utm.y()), (e_utm.x(), e_utm.y()), row['width'] / 2.0,
                row['tangent_pos'],
            )
            if ends is None:
                continue
            a = to_wgs.transform(QgsPointXY(*ends[0]))
            b = to_wgs.transform(QgsPointXY(*ends[1]))
        except Exception as exc:  # nosec B110 B112
            logger.warning(f"Tangent line for leg {row['id']} skipped: {exc}")
            continue
        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsLineString([QgsPoint(a.x(), a.y()), QgsPoint(b.x(), b.y())]))
        feat.setAttributes([f"Tangent Line {row['id']}"])
        feats.append(feat)
    if not feats:
        return None
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    try:
        sym = QgsLineSymbol.createSimple({'color': color, 'width': '0.5', 'line_style': 'dash'})
        layer.setRenderer(QgsSingleSymbolRenderer(sym))
    except Exception:  # nosec B110 B112
        pass
    return layer


# ---------------------------------------------------------------------------
# Group assembly
# ---------------------------------------------------------------------------

def add_snapshot_to_project(
    snapshot: dict[str, Any],
    group_name: str,
    *,
    gpkg_path: str | Path | None = None,
    color: str = 'red',
    run_label: str | None = None,
) -> dict[str, Any]:
    """Build the model layers (+ result layers) for ``snapshot`` and put
    them in a new layer-tree group at the top of the project.

    Returns ``{'group': QgsLayerTreeGroup, 'model': [layers], 'results': [layers]}``.
    ``results`` is empty when ``gpkg_path`` is missing or has no layers.
    """
    from qgis.core import QgsProject
    from omrat_utils.run_persistence import load_run_results_to_map

    project = QgsProject.instance()
    root = project.layerTreeRoot()
    group = root.insertGroup(0, group_name)

    def _add(layer) -> None:
        project.addMapLayer(layer, False)
        group.insertLayer(0, layer)

    model_layers: list[Any] = []
    builders = (
        lambda: build_area_layer(
            snapshot.get('depths'), 'depth', 'Depth Areas',
            fill='173,216,230,110', outline='70,130,180,255',
        ),
        lambda: build_area_layer(
            snapshot.get('objects'), 'height', 'Structures',
            fill='190,190,190,140', outline='90,90,90,255',
        ),
        lambda: build_leg_layer(snapshot.get('segment_data'), 'Legs', color=color),
        lambda: build_tangent_layer(snapshot.get('segment_data'), 'Tangent Lines', color=color),
    )
    for build in builders:
        try:
            layer = build()
        except Exception as exc:  # nosec B110 B112
            logger.warning(f"Snapshot layer skipped: {exc}")
            continue
        if layer is not None:
            _add(layer)
            model_layers.append(layer)

    result_layers: list[Any] = []
    if gpkg_path is not None and Path(gpkg_path).is_file():
        try:
            result_layers = load_run_results_to_map(
                gpkg_path, run_label or group_name, add_to_project=False,
            )
        except Exception as exc:  # nosec B110 B112
            logger.warning(f"Result layers skipped: {exc}")
            result_layers = []
        for layer in result_layers:
            _add(layer)

    return {'group': group, 'model': model_layers, 'results': result_layers}
