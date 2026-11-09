"""Speed restriction zones in QGIS: live state, map layer, drawing and loading.

The pure rules live in :mod:`compute.speed_restriction`; this module keeps
the project's zones (``omrat.speed_zones.zones``), the map layer
**Speed restriction zones** and the two ways a polygon gets in (drawn on
the canvas, loaded from a file).  The dialog
(:mod:`omrat_utils.speed_restriction_dialog`) only edits through here.

* The layer is there whenever the project has zones (the user can hide it
  in QGIS) and is rebuilt from ``zones`` after every change.  Default
  style: a thin grid for an enabled zone, a grey dashed outline for one
  that is switched off, the limit as label.  A restyle is kept in the
  ``layer_styles`` block (key ``speed_zones``).
* Vertex edits on the layer reach the zone through the layer's
  ``geometryChanged`` (never the edit buffer's -- see
  ``HandleQGISIface.wire_leg_layer``).
* Every change calls the listeners (an open dialog reloads), refreshes the
  Traffic tab's ``[speed zone]`` labels and reports new overlaps on the
  message bar.
"""
from __future__ import annotations

import copy
import json
import re
from typing import TYPE_CHECKING, Any, Callable

from qgis.PyQt.QtCore import QMetaType, QTimer
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsFillSymbol,
    QgsGeometry,
    QgsLinePatternFillSymbolLayer,
    QgsMessageLog,
    QgsPalLayerSettings,
    QgsProject,
    QgsRuleBasedRenderer,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)

from compute.speed_restriction import (
    DEFAULT_COMPLIANCE_PCT, DEFAULT_LIMIT_KN, ZONES_KEY, find_overlaps, leg_exposure, new_zone,
    next_zone_id, normalise_zones, overlap_message, zone_geometry, zone_leg_effects, zone_problems,
)
from compute.speed_restriction import blocking_problems as _blocking_problems
from compute.speed_restriction import length_class_bounds

if TYPE_CHECKING:
    from omrat import OMRAT

LAYER_NAME = 'Speed restriction zones'
DRAW_LAYER_NAME = 'New speed restriction zone (right-click to finish)'
STYLE_KEY = 'speed_zones'
ENABLED_RULE_LABEL = 'Speed restriction zone'
DISABLED_RULE_LABEL = 'Speed restriction zone (switched off)'
GRID_COLOUR = '#d95f02'
LEG_TAG = 'speed zone'

_LIMIT_FIELD_RE = re.compile(r'(speed|limit|knot|kn$|^kn|fart)', re.IGNORECASE)
_NAME_FIELD_RE = re.compile(r'^(name|namn|label|zone)$', re.IGNORECASE)


def guess_field(names: list[str], kind: str) -> str | None:
    """The attribute that most likely holds the speed limit (``kind='limit'``)
    or the zone name (``kind='name'``), or ``None``."""
    pattern = _LIMIT_FIELD_RE if kind == 'limit' else _NAME_FIELD_RE
    return next((n for n in names if pattern.search(str(n))), None)


def _float_or(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if v == v and v not in (float('inf'), float('-inf')) else default


class SpeedRestrictionZones:
    """The project's speed restriction zones and their map layer."""

    def __init__(self, omrat: "OMRAT") -> None:
        self.omrat = omrat
        self.zones: list[dict[str, Any]] = []
        self.layer: QgsVectorLayer | None = None
        self._listeners: list[Callable[[], None]] = []
        self._legs_cache: tuple[str, frozenset[str]] = ('', frozenset())
        self._draw_layer: QgsVectorLayer | None = None
        self._draw_callback: Callable[[str | None], None] | None = None
        self._refreshing = False
        self.last_messages: list[str] = []

    # ------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------

    def add_listener(self, callback: Callable[[], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _changed(self, check_zone: int | None = None, refresh_layer: bool = True) -> None:
        """After every change: layer, labels, listeners, overlap report."""
        self._legs_cache = ('', frozenset())
        if refresh_layer:
            self.refresh_layer()
        geoms = getattr(self.omrat, 'qgis_geoms', None)
        if geoms is not None and hasattr(geoms, 'refresh_traffic_link_views'):
            try:
                geoms.refresh_traffic_link_views()
            except Exception:  # nosec B110 - labels are cosmetic
                pass
        if check_zone is not None:
            messages = self.overlap_messages(check_zone)
            if messages:
                self._report(' '.join(messages), Qgis.MessageLevel.Critical)
        for callback in list(self._listeners):
            try:
                callback()
            except RuntimeError:
                self.remove_listener(callback)

    def _report(self, message: str, level=Qgis.MessageLevel.Info) -> None:
        self.last_messages.append(message)
        notifier = getattr(self.omrat, 'notifier', None)
        try:
            if notifier is not None:
                notifier.display_message(message, level=level, duration=10)
                return
        except Exception:  # nosec B110 - fall back to the log
            pass
        QgsMessageLog.logMessage(message, 'OMRAT', level)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def to_save(self) -> list[dict[str, Any]]:
        """The block written to ``.omrat`` (normalised copy)."""
        return copy.deepcopy(normalise_zones(self.zones))

    def load(self, zones: Any, merge: bool = False) -> None:
        """Replace the zones with ``zones`` (file load), or append them
        (``merge``: colliding ids get the next free one)."""
        incoming = normalise_zones(zones)
        if merge and self.zones:
            used = {z['id'] for z in self.zones}
            for zone in incoming:
                if zone['id'] in used:
                    zone['id'] = next_zone_id(used)
                used.add(zone['id'])
            self.zones = normalise_zones(self.zones + incoming)
        else:
            self.zones = incoming
        self._changed()

    def clear(self) -> None:
        self.cancel_drawing()
        self.zones = []
        self.remove_layer()
        self._changed(refresh_layer=False)

    def zone(self, zone_id: int) -> dict[str, Any] | None:
        return next((z for z in self.zones if z['id'] == int(zone_id)), None)

    def add_zone(self, wkt: str, name: str = '', limit_kn: float = DEFAULT_LIMIT_KN,
                 compliance_pct: float = DEFAULT_COMPLIANCE_PCT, notify: bool = True) -> dict[str, Any]:
        """New enabled zone that applies to every ship."""
        zone = new_zone(next_zone_id(self.zones), wkt, name=name, limit_kn=limit_kn,
                        compliance_pct=compliance_pct)
        self.zones.append(zone)
        if notify:
            self._changed(check_zone=zone['id'])
        return zone

    def remove_zone(self, zone_id: int) -> bool:
        before = len(self.zones)
        self.zones = [z for z in self.zones if z['id'] != int(zone_id)]
        if len(self.zones) == before:
            return False
        self._changed()
        return True

    _EDITABLE = ('name', 'enabled', 'limit_kn', 'compliance_pct', 'wkt', 'rules')

    def update_zone(self, zone_id: int, **fields: Any) -> dict[str, Any] | None:
        """Change fields of one zone (``name``, ``enabled``, ``limit_kn``,
        ``compliance_pct``, ``wkt``, ``rules``)."""
        zone = self.zone(zone_id)
        if zone is None:
            return None
        for key, value in fields.items():
            if key not in self._EDITABLE:
                raise KeyError(key)
            zone[key] = copy.deepcopy(value)
        clean = normalise_zones([zone])[0]
        clean['id'] = zone['id']
        zone.clear()
        zone.update(clean)
        geometry_or_state = any(k in fields for k in ('wkt', 'enabled'))
        self._changed(check_zone=zone['id'] if geometry_or_state else None)
        return zone

    def ship_categories(self) -> dict[str, Any]:
        """The project's ship types and length classes (live)."""
        try:
            from omrat_utils.gather_data import GatherData
            return GatherData(self.omrat).get_ship_categories_for_save()
        except Exception:  # noqa: BLE001 - no ship categories yet
            return {'types': [], 'length_intervals': []}

    def _project(self) -> dict[str, Any]:
        return {
            'segment_data': getattr(self.omrat, 'segment_data', None) or {},
            'traffic_data': getattr(self.omrat, 'traffic_data', None) or {},
            'ship_categories': self.ship_categories(),
            ZONES_KEY: self.zones,
        }

    def problems(self) -> dict[int, list[str]]:
        """``{zone_id: [problem, ...]}`` for every zone with one (overlaps
        count only between enabled zones)."""
        cats = self.ship_categories()
        types = list(cats.get('types') or []) or None
        bounds = length_class_bounds(cats.get('length_intervals'))
        out: dict[int, list[str]] = {}
        for zone in self.zones:
            found = zone_problems(zone, types, bounds)
            if found:
                out[zone['id']] = found
        by_id = {z['id']: z for z in self.zones}
        for a, b, area in find_overlaps(self.zones):
            msg = overlap_message(by_id[a], by_id[b], area)
            out.setdefault(a, []).append(msg)
            out.setdefault(b, []).append(msg)
        return out

    def overlap_messages(self, zone_id: int) -> list[str]:
        """Overlaps of one zone with the other enabled zones."""
        by_id = {z['id']: z for z in self.zones}
        return [overlap_message(by_id[a], by_id[b], area)
                for a, b, area in find_overlaps(self.zones) if int(zone_id) in (a, b)]

    def blocking_problems(self) -> list[str]:
        """What stops Run model (see :func:`compute.speed_restriction.blocking_problems`)."""
        return _blocking_problems(self._project())

    def leg_effects(self, zone: dict[str, Any]) -> list[dict[str, Any]]:
        """The legs one zone touches and its effect on their speeds."""
        return zone_leg_effects(self._project(), zone)

    def zone_legs(self) -> frozenset[str]:
        """Legs touched by an enabled zone (cached on zones + endpoints)."""
        segs = getattr(self.omrat, 'segment_data', None) or {}
        enabled = [z for z in self.zones if z.get('enabled', True)]
        if not enabled or not segs:
            return frozenset()
        sig = json.dumps([enabled, {k: (v.get('Start_Point'), v.get('End_Point'))
                                    for k, v in segs.items() if isinstance(v, dict)}],
                         sort_keys=True, default=str)
        if self._legs_cache[0] == sig:
            return self._legs_cache[1]
        geoms = [(z['id'], zone_geometry(z)) for z in enabled]
        legs = frozenset(str(k) for k, v in segs.items()
                         if isinstance(v, dict) and leg_exposure(v, geoms) is not None)
        self._legs_cache = (sig, legs)
        return legs

    def leg_tags(self, seg: str) -> list[str]:
        """``['speed zone']`` for a leg in an enabled zone (Traffic tab label)."""
        return [LEG_TAG] if str(seg) in self.zone_legs() else []

    # ------------------------------------------------------------------
    # Map layer
    # ------------------------------------------------------------------

    def layer_in_project(self) -> bool:
        layer = self.layer
        if layer is None:
            return False
        try:
            return QgsProject.instance().mapLayer(layer.id()) is not None
        except RuntimeError:
            return False

    def ensure_layer(self) -> QgsVectorLayer:
        """The zones layer, created (and styled) when missing."""
        if self.layer_in_project():
            return self.layer
        self.layer = None
        layer = QgsVectorLayer("Polygon?crs=EPSG:4326", LAYER_NAME, "memory")
        fields = QgsFields()
        fields.append(QgsField('id', QMetaType.Type.Int))
        fields.append(QgsField('name', QMetaType.Type.QString))
        fields.append(QgsField('limit_kn', QMetaType.Type.Double))
        fields.append(QgsField('compliance', QMetaType.Type.Double))
        fields.append(QgsField('enabled', QMetaType.Type.Int))
        fields.append(QgsField('rules', QMetaType.Type.Int))
        layer.dataProvider().addAttributes(fields.toList())
        layer.updateFields()
        self.apply_default_style(layer)
        try:
            from omrat_utils.layer_styles import apply_stored_style
            apply_stored_style(self.omrat, STYLE_KEY, layer)
        except Exception:  # nosec B110 - default style stays
            pass
        QgsProject.instance().addMapLayer(layer)
        layer.geometryChanged.connect(self._on_geometry_changed)
        self.layer = layer
        return layer

    @staticmethod
    def apply_default_style(layer: QgsVectorLayer) -> None:
        """Thin grid for enabled zones, grey dashed outline when off, limit label."""
        grid = QgsFillSymbol.createSimple({'style': 'no', 'outline_color': GRID_COLOUR, 'outline_width': '0.4'})
        for angle in (0.0, 90.0):
            pattern = QgsLinePatternFillSymbolLayer()
            pattern.setLineAngle(angle)
            pattern.setDistance(3.0)
            pattern.setLineWidth(0.15)
            pattern.setColor(QColor(GRID_COLOUR))
            grid.insertSymbolLayer(0, pattern)
        off = QgsFillSymbol.createSimple({'style': 'no', 'outline_color': '#8c8c8c', 'outline_width': '0.4',
                                          'outline_style': 'dash'})
        root = QgsRuleBasedRenderer.Rule(None)
        root.appendChild(QgsRuleBasedRenderer.Rule(grid, 0, 0, '"enabled" = 1', ENABLED_RULE_LABEL))
        root.appendChild(QgsRuleBasedRenderer.Rule(off, 0, 0, '"enabled" = 0', DISABLED_RULE_LABEL))
        layer.setRenderer(QgsRuleBasedRenderer(root))
        settings = QgsPalLayerSettings()
        settings.fieldName = "\"name\" || '\\n' || round(\"limit_kn\", 1) || ' kn'"
        settings.isExpression = True
        settings.enabled = True
        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
        layer.setLabelsEnabled(True)

    def refresh_layer(self) -> None:
        """Rebuild the features from ``zones`` (the layer goes when there are none)."""
        if not self.zones:
            self.remove_layer()
            return
        layer = self.ensure_layer()
        was_editing = layer.isEditable()
        self._refreshing = True
        try:
            if was_editing:
                # Vertex edits already reached the zones (geometryChanged).
                layer.commitChanges()
            provider = layer.dataProvider()
            provider.truncate()
            feats = []
            for zone in self.zones:
                geom = QgsGeometry.fromWkt(zone.get('wkt') or '')
                if geom is None or geom.isNull() or geom.isEmpty():
                    continue
                feat = QgsFeature(layer.fields())
                feat.setGeometry(geom)
                feat.setAttributes([int(zone['id']), str(zone['name']), float(zone['limit_kn']),
                                    float(zone['compliance_pct']), 1 if zone.get('enabled', True) else 0,
                                    len(zone.get('rules') or [])])
                feats.append(feat)
            provider.addFeatures(feats)
            layer.updateExtents()
            if was_editing:
                layer.startEditing()
            layer.triggerRepaint()
        finally:
            self._refreshing = False

    def remove_layer(self) -> None:
        layer, self.layer = self.layer, None
        if layer is None:
            return
        try:
            layer.geometryChanged.disconnect(self._on_geometry_changed)
        except (TypeError, RuntimeError):
            pass
        try:
            if layer.isEditable():
                layer.rollBack()
            if QgsProject.instance().mapLayer(layer.id()) is not None:
                QgsProject.instance().removeMapLayer(layer.id())
        except RuntimeError:
            pass

    def _on_geometry_changed(self, fid: int, geom: QgsGeometry) -> None:
        """A vertex edit on the layer: store the new polygon on its zone."""
        if self._refreshing or self.layer is None:
            return
        try:
            zone_id = int(self.layer.getFeature(fid)['id'])
        except Exception:  # noqa: BLE001 - not one of our features
            return
        zone = self.zone(zone_id)
        if zone is None:
            return
        zone['wkt'] = geom.asWkt(precision=6)
        self._changed(check_zone=zone_id, refresh_layer=False)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def is_drawing(self) -> bool:
        return self._draw_layer is not None

    def start_drawing(self, on_done: Callable[[str | None], None]) -> bool:
        """Let the user digitise one polygon; ``on_done(wkt)`` gets it in
        EPSG:4326 (``None`` when cancelled)."""
        self.cancel_drawing()
        iface = getattr(self.omrat, 'iface', None)
        layer = QgsVectorLayer("Polygon?crs=EPSG:4326", DRAW_LAYER_NAME, "memory")
        QgsProject.instance().addMapLayer(layer)
        layer.startEditing()
        layer.featureAdded.connect(self._on_draw_feature_added)
        self._draw_layer = layer
        self._draw_callback = on_done
        try:
            if iface is not None:
                iface.setActiveLayer(layer)
                iface.actionAddFeature().trigger()
        except Exception:  # nosec B110 - headless: the caller finishes the drawing
            pass
        return True

    def _on_draw_feature_added(self, fid: int) -> None:
        layer = self._draw_layer
        if layer is None:
            return
        try:
            geom = layer.getFeature(fid).geometry()
            wkt = geom.asWkt(precision=6) if geom is not None and not geom.isEmpty() else None
        except Exception:  # noqa: BLE001 - treated as cancelled
            wkt = None
        # Removing the layer inside its own signal is unsafe: defer.
        if getattr(self.omrat, 'testing', False):
            self.finish_drawing(wkt)
        else:
            QTimer.singleShot(0, lambda: self.finish_drawing(wkt))

    def finish_drawing(self, wkt: str | None) -> None:
        """End the drawing and hand the polygon to the callback."""
        callback = self._draw_callback
        self._draw_callback = None
        self._drop_draw_layer()
        if callback is not None:
            callback(wkt)

    def cancel_drawing(self) -> None:
        if self._draw_layer is None:
            return
        callback = self._draw_callback
        self._draw_callback = None
        self._drop_draw_layer()
        if callback is not None:
            callback(None)

    def _drop_draw_layer(self) -> None:
        layer, self._draw_layer = self._draw_layer, None
        if layer is None:
            return
        try:
            layer.featureAdded.disconnect(self._on_draw_feature_added)
        except (TypeError, RuntimeError):
            pass
        try:
            if layer.isEditable():
                layer.rollBack()
            if QgsProject.instance().mapLayer(layer.id()) is not None:
                QgsProject.instance().removeMapLayer(layer.id())
        except RuntimeError:
            pass
        iface = getattr(self.omrat, 'iface', None)
        try:
            if iface is not None:
                iface.actionPan().trigger()
        except Exception:  # nosec B110 - the map tool is cosmetic
            pass

    # ------------------------------------------------------------------
    # Loading from a file
    # ------------------------------------------------------------------

    def add_from_layer(self, layer: QgsVectorLayer, limit_field: str | None = None,
                       name_field: str | None = None,
                       default_limit: float = DEFAULT_LIMIT_KN) -> list[dict[str, Any]]:
        """One zone per polygon feature of ``layer`` (reprojected to EPSG:4326).

        The limit comes from ``limit_field`` (``default_limit`` when empty
        or not a number), the name from ``name_field``.  Non-polygon
        features are skipped.
        """
        tr = None
        src = layer.crs()
        if src.isValid() and src.authid() != 'EPSG:4326':
            tr = QgsCoordinateTransform(src, QgsCoordinateReferenceSystem('EPSG:4326'),
                                        QgsProject.instance().transformContext())
        names = layer.fields().names()
        added: list[dict[str, Any]] = []
        for feat in layer.getFeatures():
            geom = QgsGeometry(feat.geometry())
            if geom.isNull() or geom.isEmpty():
                continue
            if tr is not None:
                geom.transform(tr)
            wkt = geom.asWkt(precision=6)
            if zone_geometry({'wkt': wkt}) is None:
                continue
            limit = _float_or(feat[limit_field], default_limit) if limit_field in names else default_limit
            name = str(feat[name_field] or '') if name_field in names else ''
            added.append(self.add_zone(wkt, name=name, limit_kn=limit if limit > 0 else default_limit,
                                       notify=False))
        if added:
            self._changed()
            messages = []
            for zone in added:
                messages.extend(m for m in self.overlap_messages(zone['id']) if m not in messages)
            if messages:
                self._report(' '.join(messages), Qgis.MessageLevel.Critical)
        return added

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------

    def unload(self) -> None:
        self.cancel_drawing()
        self.remove_layer()
        self._listeners.clear()


__all__ = [
    'SpeedRestrictionZones', 'LAYER_NAME', 'DRAW_LAYER_NAME', 'STYLE_KEY', 'LEG_TAG', 'guess_field',
    'ENABLED_RULE_LABEL', 'DISABLED_RULE_LABEL',
]
