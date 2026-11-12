"""**Depths > Simplify...** -- a lighter depth layer for the drift model.

The drift model works through every depth class for each leg, direction
and wind sector, so a detailed depth layer makes a run slow.  The tool
keeps every class down to a detail limit (suggested from the deepest
draught in the traffic) in the whole area, and every class in the drift
corridors that reach a structure: the drift model's own corridor for each
leg, travel direction and drift direction, cut just past the structure.
Deeper water outside them is removed.  See :mod:`geometries.depth_simplify`
for why that keeps grounding and allision unchanged.

``apply_simplification`` / ``restore_original`` are the headless cores;
the dialog only collects the limit and the corridor length.  The depth
areas in use before the first simplification are kept as the layer
:data:`BEFORE_LAYER_NAME` and in memory, so **Restore original** puts
them back -- also after a second simplification with other settings,
which always starts from the original again.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qgis.PyQt.QtCore import QMetaType, QTimer
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFillSymbol,
    QgsGeometry,
    QgsProject,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
)

from geometries.depth_simplify import (
    DEFAULT_REPAIR_QUANTILE,
    NM_M,
    REPAIR_QUANTILES,
    area_km2,
    class_values,
    corridor_length_m,
    corridor_zone,
    depth_value,
    describe,
    drift_lines,
    max_traffic_draught,
    simplify_depths,
    suggest_detail_limit,
)

if TYPE_CHECKING:
    from omrat import OMRAT

BEFORE_LAYER_NAME = 'Depth Areas (before simplification)'
ZONE_LAYER_NAME = 'Depth simplification zone'
FALLBACK_LENGTH_NM = 5.0
"""Corridor length offered when the drift settings give no repair time / speed."""
ZONE_REFRESH_MS = 300
"""Delay before the map outline follows a typed corridor length."""


# ---------------------------------------------------------------------------
# Headless core
# ---------------------------------------------------------------------------

def structure_wkts(omrat: "OMRAT") -> list[str]:
    return [row[2] for row in omrat.object.object_rows()]


def corridor_length(omrat: "OMRAT", quantile: float) -> float | None:
    """Corridor length for ``quantile`` (the model reach for 99 %) from the drift settings."""
    try:
        return corridor_length_m(getattr(omrat, 'drift_values', None), getattr(omrat, 'segment_data', None),
                                 quantile)
    except Exception:  # nosec B110 - an unusable repair setting just gives no suggestion
        return None


def build_zone(omrat: "OMRAT", length_m: float) -> tuple[Any, list[dict[str, Any]]]:
    """``(zone, corridors)``: the drift corridors of ``length_m`` that reach a structure."""
    drift = getattr(omrat, 'drift_values', None) or {}
    lines = drift_lines(getattr(omrat, 'segment_data', None) or {}, getattr(omrat, 'traffic_data', None))
    return corridor_zone(lines, structure_wkts(omrat), length_m, drift.get('rose'))


def suggested_limit(omrat: "OMRAT", rows: list[list[str]]) -> tuple[float | None, float | None]:
    """``(detail limit, deepest draught)`` suggested from the traffic (squat included)."""
    names = None
    try:
        names = omrat.drift_settings._project_type_names()
    except Exception:  # nosec B110 -- no settings handler (tests): index-based block coefficients
        names = None
    from compute.squat import KEY_DRIFTING, KEY_POWERED, squat_on
    pc = getattr(getattr(omrat, 'causation_f', None), 'data', None)
    with_squat = squat_on(pc, KEY_DRIFTING) or squat_on(pc, KEY_POWERED)
    draught = max_traffic_draught(getattr(omrat, 'traffic_data', None), with_squat, names)
    return suggest_detail_limit(class_values(rows), draught), draught


def _state(omrat: "OMRAT") -> dict[str, Any] | None:
    return getattr(omrat, '_depth_simplify_state', None)


def source_rows(omrat: "OMRAT") -> list[list[str]]:
    """The rows to simplify: the original when the table still holds our last result."""
    current = omrat.object.depth_rows()
    state = _state(omrat)
    if state and current == state['result']:
        return state['original']
    return current


def is_simplified(omrat: "OMRAT") -> bool:
    """True while the depth table is the result of the last simplification."""
    state = _state(omrat)
    return bool(state) and omrat.object.depth_rows() == state['result']


def apply_simplification(omrat: "OMRAT", detail_limit: float, length_m: float) -> dict[str, Any]:
    """Replace the depth areas with the simplified ones; returns the summary."""
    original = source_rows(omrat)
    if not original:
        raise ValueError('There are no depth areas to simplify.')
    zone, corridors = build_zone(omrat, length_m)
    rows, summary = simplify_depths(original, detail_limit, zone)
    summary['corridor_length_m'] = float(length_m)
    summary['structures'] = len(structure_wkts(omrat))
    summary['corridors'] = len(corridors)
    summary['legs'] = len({c['segment'] for c in corridors})
    summary['zone_km2'] = area_km2(zone)
    show_before_layer(omrat, original)
    omrat.object.replace_depths(rows)
    omrat._depth_simplify_state = {'original': original, 'result': omrat.object.depth_rows()}
    return summary


def restore_original(omrat: "OMRAT") -> bool:
    """Put the depth areas from before the simplification back; False if they changed since."""
    if not is_simplified(omrat):
        return False
    omrat.object.replace_depths(_state(omrat)['original'])
    omrat._depth_simplify_state = None
    remove_layer(BEFORE_LAYER_NAME)
    return True


# ---------------------------------------------------------------------------
# Map layers
# ---------------------------------------------------------------------------

def remove_layer(name: str) -> None:
    project = QgsProject.instance()
    for layer in project.mapLayersByName(name):
        project.removeMapLayer(layer.id())


def _hide_in_tree(layer: QgsVectorLayer) -> None:
    node = QgsProject.instance().layerTreeRoot().findLayer(layer.id())
    if node is not None:
        node.setItemVisibilityChecked(False)


def show_before_layer(omrat: "OMRAT", rows: list[list[str]]) -> QgsVectorLayer:
    """(Re)create the layer with the depth areas from before the simplification, hidden."""
    remove_layer(BEFORE_LAYER_NAME)
    layer = QgsVectorLayer('Polygon?crs=epsg:4326', BEFORE_LAYER_NAME, 'memory')
    pr = layer.dataProvider()
    pr.addAttributes([QgsField('id', QMetaType.Type.Int), QgsField('depth', QMetaType.Type.Double)])
    layer.updateFields()
    feats = []
    for i, (_, depth, wkt) in enumerate(rows, start=1):
        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsGeometry.fromWkt(wkt))
        feat.setAttributes([i, depth_value(depth)])
        feats.append(feat)
    pr.addFeatures(feats)
    layer.updateExtents()
    depth_layer = getattr(omrat.object, 'depth_layer', None)
    try:
        if depth_layer is not None and depth_layer.renderer() is not None:
            layer.setRenderer(depth_layer.renderer().clone())
    except RuntimeError:
        pass
    QgsProject.instance().addMapLayer(layer)
    _hide_in_tree(layer)
    return layer


def show_zone_layer(zone) -> QgsVectorLayer | None:
    """Dashed outline of the zone (replaces the previous one)."""
    remove_layer(ZONE_LAYER_NAME)
    if zone is None or zone.is_empty:
        return None
    layer = QgsVectorLayer('Polygon?crs=epsg:4326', ZONE_LAYER_NAME, 'memory')
    feat = QgsFeature()
    feat.setGeometry(QgsGeometry.fromWkt(zone.wkt))
    layer.dataProvider().addFeatures([feat])
    layer.updateExtents()
    symbol = QgsFillSymbol.createSimple({
        'color': '0,0,0,0', 'outline_color': '176,23,123', 'outline_width': '0.6', 'outline_style': 'dash',
    })
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))
    QgsProject.instance().addMapLayer(layer)
    return layer


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

def _quantile_label(q: float) -> str:
    pct = f'{100 * q:.10g} %'
    if q == DEFAULT_REPAIR_QUANTILE:
        return f'Model reach ({pct} repaired)'
    return f'{pct} repaired (margin)'


class DepthSimplifyDialog(QDialog):
    """Modeless: the zone is drawn on the map while the dialog is open."""

    CUSTOM = -1.0

    def __init__(self, omrat: "OMRAT", parent=None) -> None:
        super().__init__(parent)
        self.omrat = omrat
        self.setWindowTitle('Simplify depth areas')
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)

        intro = QLabel(
            'The drift model works through every depth class, so fewer depth areas give a faster run. '
            'All classes down to the detail limit are kept everywhere (grounding and anchoring). '
            'Deeper classes are kept only in the drift corridors that reach a structure, where ships may '
            'anchor before they hit it; elsewhere they are removed.')
        intro.setWordWrap(True)
        layout.addWidget(intro)

        detail = QGroupBox('Everywhere')
        form = QFormLayout(detail)
        self.cbLimit = QComboBox()
        form.addRow('Keep all classes down to', self.cbLimit)
        self.lblLimitHint = QLabel()
        self.lblLimitHint.setWordWrap(True)
        form.addRow(self.lblLimitHint)
        layout.addWidget(detail)

        zone = QGroupBox('Drift corridors that reach a structure (all classes kept)')
        form = QFormLayout(zone)
        self.cbQuantile = QComboBox()
        for q in REPAIR_QUANTILES:
            self.cbQuantile.addItem(_quantile_label(q), q)
        self.cbQuantile.addItem('Custom length', self.CUSTOM)
        form.addRow('Corridor length', self.cbQuantile)
        self.sbLength = QDoubleSpinBox()
        self.sbLength.setRange(0.0, 500.0)
        self.sbLength.setDecimals(1)
        self.sbLength.setSuffix(' nm')
        form.addRow('', self.sbLength)
        self.lblZoneHint = QLabel()
        self.lblZoneHint.setWordWrap(True)
        form.addRow(self.lblZoneHint)
        layout.addWidget(zone)

        self.lblResult = QLabel()
        self.lblResult.setWordWrap(True)
        layout.addWidget(self.lblResult)

        buttons = QHBoxLayout()
        self.pbRestore = QPushButton('Restore original')
        self.pbSimplify = QPushButton('Simplify')
        self.pbClose = QPushButton('Close')
        buttons.addWidget(self.pbRestore)
        buttons.addStretch(1)
        buttons.addWidget(self.pbSimplify)
        buttons.addWidget(self.pbClose)
        layout.addLayout(buttons)

        self._reach_known = True
        self._zone_timer = QTimer(self)
        self._zone_timer.setSingleShot(True)
        self._zone_timer.setInterval(ZONE_REFRESH_MS)
        self._zone_timer.timeout.connect(self.refresh_zone)

        self.cbQuantile.currentIndexChanged.connect(self._on_quantile)
        self.sbLength.valueChanged.connect(self._on_length)
        self.pbSimplify.clicked.connect(self.on_simplify)
        self.pbRestore.clicked.connect(self.on_restore)
        self.pbClose.clicked.connect(self.close)
        self.reload()

    # -- state ------------------------------------------------------------

    def reload(self) -> None:
        """Fill the controls from the project (classes, traffic, drift settings)."""
        rows = source_rows(self.omrat)
        classes = class_values(rows)
        limit, draught = suggested_limit(self.omrat, rows)
        self.cbLimit.clear()
        for c in classes:
            self.cbLimit.addItem(f'{c:g} m', c)
        if limit is not None:
            self.cbLimit.setCurrentIndex(classes.index(limit))
        if draught is None:
            self.lblLimitHint.setText('No draughts in the traffic: the deepest class is suggested.')
        else:
            self.lblLimitHint.setText(
                f'Suggested {limit:g} m: the first class deeper than the deepest draught in the traffic '
                f'({draught:.1f} m, squat included).  A class value is the shallow edge of its band.')
        self.cbQuantile.blockSignals(True)
        self.cbQuantile.setCurrentIndex(REPAIR_QUANTILES.index(DEFAULT_REPAIR_QUANTILE))
        self.cbQuantile.blockSignals(False)
        self._on_quantile()
        self.pbSimplify.setEnabled(bool(classes))
        self.pbRestore.setEnabled(is_simplified(self.omrat))

    def detail_limit(self) -> float | None:
        value = self.cbLimit.currentData()
        return None if value is None else float(value)

    def length_m(self) -> float:
        return self.sbLength.value() * NM_M

    def _on_quantile(self, *_args) -> None:
        q = self.cbQuantile.currentData()
        if q is None or q == self.CUSTOM:
            self.refresh_zone()
            return
        length = corridor_length(self.omrat, q)
        self._reach_known = length is not None
        self.sbLength.blockSignals(True)
        self.sbLength.setValue((length if length is not None else FALLBACK_LENGTH_NM * NM_M) / NM_M)
        self.sbLength.blockSignals(False)
        self.refresh_zone()

    def _on_length(self, *_args) -> None:
        if self.cbQuantile.currentData() != self.CUSTOM:
            self.cbQuantile.blockSignals(True)
            self.cbQuantile.setCurrentIndex(self.cbQuantile.count() - 1)
            self.cbQuantile.blockSignals(False)
        self._reach_known = True
        self._zone_timer.start()

    def refresh_zone(self) -> None:
        """Redraw the corridor outline and its description for the current length."""
        self._zone_timer.stop()
        n_struct = len(structure_wkts(self.omrat))
        zone, corridors = build_zone(self.omrat, self.length_m())
        show_zone_layer(zone)
        if not n_struct:
            text = 'There are no structures, so every class deeper than the limit is removed.'
        elif not corridors:
            text = 'No drift corridor reaches a structure, so every class deeper than the limit is removed.'
        else:
            legs = len({c['segment'] for c in corridors})
            text = (f'{len(corridors)} drift corridor(s) from {legs} leg(s) reach the {n_struct} structure(s): '
                    f'{area_km2(zone):,.0f} km² (dashed on the map).  The drift model reads nothing '
                    'beyond its reach, so a longer corridor only adds margin.')
        if not self._reach_known:
            text += '  Drift settings give no repair time and speed, so the length is a default.'
        self.lblZoneHint.setText(text)

    # -- actions ----------------------------------------------------------

    def _confirm(self, limit: float) -> bool:
        if getattr(self.omrat, 'testing', False):
            return True
        answer = QMessageBox.question(
            self, 'Simplify depth areas',
            f'Replace the depth areas?\n\nAll classes down to {limit:g} m are kept everywhere, deeper ones only '
            f'in the drift corridors ({self.sbLength.value():.1f} nm) that reach a structure.\n\n'
            f'The current depth areas are kept as the hidden layer "{BEFORE_LAYER_NAME}", and Restore '
            'original puts them back while QGIS is open.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Yes

    def on_simplify(self) -> dict[str, Any] | None:
        limit = self.detail_limit()
        if limit is None or not self._confirm(limit):
            return None
        try:
            summary = apply_simplification(self.omrat, limit, self.length_m())
        except ValueError as exc:
            self.lblResult.setText(str(exc))
            return None
        message = describe(summary)
        self.lblResult.setText(message + '.')
        self.pbRestore.setEnabled(True)
        self._notify(message)
        return summary

    def on_restore(self) -> bool:
        ok = restore_original(self.omrat)
        if ok:
            self.lblResult.setText('The depth areas from before the simplification are back.')
        else:
            self.lblResult.setText('The depth areas have changed since they were simplified; nothing restored.')
        self.pbRestore.setEnabled(False)
        return ok

    def _notify(self, message: str) -> None:
        notifier = getattr(self.omrat, 'notifier', None)
        if notifier is None:
            return
        try:
            notifier.display_message(message, duration=10)
        except Exception:  # nosec B110 B112
            pass

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._zone_timer.stop()
        remove_layer(ZONE_LAYER_NAME)
        super().closeEvent(event)


def open_depth_simplify(omrat: "OMRAT") -> DepthSimplifyDialog | None:
    """Depths tab button: open the dialog (modeless) or raise the open one."""
    existing = getattr(omrat, '_depth_simplify_dlg', None)
    if existing is not None:
        try:
            existing.reload()
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return existing
        except RuntimeError:
            omrat._depth_simplify_dlg = None
    if not omrat.object.depth_rows():
        QMessageBox.information(getattr(omrat, 'main_widget', None), 'Simplify depth areas',
                                'There are no depth areas yet. Load or download depths first.')
        return None
    dlg = DepthSimplifyDialog(omrat, getattr(omrat, 'main_widget', None))
    # Keep a Python reference, otherwise the modeless dialog is collected.
    omrat._depth_simplify_dlg = dlg
    dlg.show()
    return dlg
