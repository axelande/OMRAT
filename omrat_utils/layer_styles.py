"""Persist QGIS layer styling in the ``.omrat`` project file.

All OMRAT layers are memory layers rebuilt from the project file, so a
colour or line width changed in the QGIS Layers panel used to be lost on
the next Load.  This module round-trips the QGIS *named style* (the QML
XML) for each layer type through a ``layer_styles`` block:

``{"legs": "<qgis ...>", "tangent": ..., "depths": ..., "structures": ...}``

* **legs** -- every leg is its own layer; the first leg layer's style is
  captured and applied to all of them.
* **tangent** -- the single *Tangent Line* layer.
* **depths** -- the consolidated depth layer.  Note that OMRAT re-applies
  its automatic depth ramp when depth intervals change, so a manual depth
  style survives Load but not an interval edit.
* **structures** -- one layer per structure; the first one's style is
  applied to all.

Only symbology, labeling, rendering and legend categories are exported,
so per-layer custom properties (for example the ``segment_id`` a drawn
leg carries) are never copied from one layer to another.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omrat import OMRAT

logger = logging.getLogger(__name__)

STYLE_KEYS: tuple[str, ...] = ('legs', 'tangent', 'depths', 'structures')
TANGENT_LAYER_NAME = 'Tangent Line'


# ---------------------------------------------------------------------------
# QML <-> layer
# ---------------------------------------------------------------------------

def _style_categories() -> Any:
    """Symbology | Labeling | Rendering | Legend, across the QGIS 3 / 4 enum
    locations; ``None`` means "use the API default" (all categories)."""
    try:
        from qgis.core import QgsMapLayer
        cat = QgsMapLayer.StyleCategory
        return cat.Symbology | cat.Labeling | cat.Rendering | cat.Legend
    except Exception:  # nosec B110 B112
        pass
    try:
        from qgis.core import Qgis
        cat = Qgis.MapLayerStyleCategory
        return cat.Symbology | cat.Labeling | cat.Rendering | cat.Legend
    except Exception:  # nosec B110 B112
        return None


def export_style(layer: Any) -> str | None:
    """QML text for ``layer``'s current style, or ``None`` on failure."""
    if layer is None:
        return None
    try:
        from qgis.core import QgsReadWriteContext
        from qgis.PyQt.QtXml import QDomDocument
        doc = QDomDocument()
        cats = _style_categories()
        if cats is not None:
            err = layer.exportNamedStyle(doc, QgsReadWriteContext(), cats)
        else:
            err = layer.exportNamedStyle(doc)
        if err:
            logger.warning(f"exportNamedStyle failed for {layer.name()!r}: {err}")
            return None
        text = doc.toString()
        return text if text and text.strip() else None
    except Exception as exc:  # nosec B110 B112
        logger.warning(f"Could not export style: {exc}")
        return None


def apply_style(layer: Any, qml: str | None) -> bool:
    """Apply QML text to ``layer``; True when QGIS accepted it."""
    if layer is None or not isinstance(qml, str) or not qml.strip():
        return False
    try:
        from qgis.PyQt.QtXml import QDomDocument
        doc = QDomDocument()
        parsed = doc.setContent(qml)
        ok_parse = parsed[0] if isinstance(parsed, tuple) else bool(parsed)
        if not ok_parse:
            logger.warning(f"Stored style for {layer.name()!r} is not valid XML")
            return False
        cats = _style_categories()
        if cats is not None:
            result = layer.importNamedStyle(doc, cats)
        else:
            result = layer.importNamedStyle(doc)
        ok = result[0] if isinstance(result, tuple) else bool(result)
        if not ok:
            msg = result[1] if isinstance(result, tuple) and len(result) > 1 else ''
            logger.warning(f"importNamedStyle failed for {layer.name()!r}: {msg}")
            return False
        layer.triggerRepaint()
        return True
    except Exception as exc:  # nosec B110 B112
        logger.warning(f"Could not apply style: {exc}")
        return False


# ---------------------------------------------------------------------------
# Locating OMRAT's layers
# ---------------------------------------------------------------------------

def _in_project(layer: Any) -> bool:
    try:
        from qgis.core import QgsProject
        return layer is not None and QgsProject.instance().mapLayer(layer.id()) is not None
    except Exception:  # nosec B110 B112
        return False


def leg_layers(omrat: "OMRAT") -> list[Any]:
    geoms = getattr(omrat, 'qgis_geoms', None)
    if geoms is None:
        return []
    tangent = getattr(geoms, 'tangent_layer', None)
    out = []
    for layer in list(getattr(geoms, 'vector_layers', []) or []):
        try:
            if layer is tangent or layer.name() == TANGENT_LAYER_NAME:
                continue
        except RuntimeError:
            continue
        if _in_project(layer):
            out.append(layer)
    return out


def tangent_layer(omrat: "OMRAT") -> Any | None:
    geoms = getattr(omrat, 'qgis_geoms', None)
    layer = getattr(geoms, 'tangent_layer', None) if geoms is not None else None
    return layer if _in_project(layer) else None


def depth_layer(omrat: "OMRAT") -> Any | None:
    obj = getattr(omrat, 'object', None)
    layer = getattr(obj, 'depth_layer', None) if obj is not None else None
    return layer if _in_project(layer) else None


def structure_layers(omrat: "OMRAT") -> list[Any]:
    obj = getattr(omrat, 'object', None)
    if obj is None:
        return []
    layers = list(getattr(obj, 'loaded_object_areas', []) or [])
    return [lyr for lyr in layers if _in_project(lyr)]


def _layers_for(omrat: "OMRAT", key: str) -> list[Any]:
    if key == 'legs':
        return leg_layers(omrat)
    if key == 'tangent':
        lyr = tangent_layer(omrat)
        return [lyr] if lyr is not None else []
    if key == 'depths':
        lyr = depth_layer(omrat)
        return [lyr] if lyr is not None else []
    if key == 'structures':
        return structure_layers(omrat)
    return []


# ---------------------------------------------------------------------------
# Project-level collect / apply
# ---------------------------------------------------------------------------

def collect_styles(omrat: "OMRAT") -> dict[str, str]:
    """Current style of each layer type that exists, as ``{key: qml}``.

    Layer types with no layer on the map keep whatever the project last
    stored (``omrat.layer_styles``) so a style is not dropped just because
    the user has not drawn a structure yet.
    """
    styles: dict[str, str] = {}
    remembered = getattr(omrat, 'layer_styles', None)
    if isinstance(remembered, dict):
        styles.update({k: v for k, v in remembered.items() if isinstance(v, str) and v})
    for key in STYLE_KEYS:
        try:
            layers = _layers_for(omrat, key)
        except Exception:  # nosec B110 B112
            layers = []
        if not layers:
            continue
        qml = export_style(layers[0])
        if qml:
            styles[key] = qml
    return styles


def apply_styles(omrat: "OMRAT", styles: dict[str, Any] | None) -> dict[str, int]:
    """Remember ``styles`` on the plugin and apply them to every matching
    layer on the map.  Returns ``{key: number_of_layers_styled}``."""
    clean = {k: v for k, v in (styles or {}).items() if k in STYLE_KEYS and isinstance(v, str) and v.strip()}
    try:
        omrat.layer_styles = dict(clean)
    except Exception:  # nosec B110 B112
        pass
    applied: dict[str, int] = {}
    for key, qml in clean.items():
        n = 0
        for layer in _layers_for(omrat, key):
            if apply_style(layer, qml):
                n += 1
        applied[key] = n
    return applied


def apply_stored_style(omrat: "OMRAT", key: str, layer: Any) -> bool:
    """Give a freshly created layer the project's stored style for ``key``
    (a leg drawn after Load looks like the loaded ones)."""
    styles = getattr(omrat, 'layer_styles', None)
    if not isinstance(styles, dict):
        return False
    return apply_style(layer, styles.get(key))
