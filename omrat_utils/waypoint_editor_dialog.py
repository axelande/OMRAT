"""**Settings > Waypoint editor...** -- connect / disconnect leg ends.

Pick a waypoint (list or map) to see the legs on it.  **Detach** moves
one leg end back along its own leg onto a new waypoint -- the shortened
leg is part of the old one, so it cannot cross the legs that stay -- by
at least the close-waypoint threshold, so the next *Update all
distributions* does not offer to join it again.  **Connect** moves a leg
end onto the waypoint; the legs on its old waypoint stay put.  Both are
permanent (no undo) and ask first.

``apply_detach`` / ``apply_attach`` are the headless cores (edit,
refresh the leg on the map and in the route table, rebuild the junction
registry while keeping hand-edited matrices -- see
:func:`geometries.junctions.carry_user_matrices`); the dialog only
collects the choice.  The pure operations live in
:mod:`geometries.waypoints`.
"""
from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

from qgis.PyQt.QtCore import QEvent, QMetaType, Qt
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
    QgsWkbTypes,
)
from qgis.gui import QgsRubberBand

from geometries.junctions import Junction, carry_user_matrices, junction_id_for_point
from geometries.route_validation import crossings_of_leg
from geometries.waypoints import (
    WaypointEditError,
    attach_candidates,
    attach_leg_end,
    detach_leg_end,
    detach_limits,
    detach_point,
    incident_legs,
    rebuild_waypoints,
)
from omrat_utils.traffic_links import leg_name

if TYPE_CHECKING:
    from omrat import OMRAT


# ---------------------------------------------------------------------------
# Headless core
# ---------------------------------------------------------------------------


def sync_waypoints(omrat: "OMRAT") -> dict:
    """``omrat.waypoints`` brought in step with ``segment_data``
    (coordinates win, ids of unmoved nodes kept), updated in place."""
    wps = getattr(omrat, 'waypoints', None)
    if not isinstance(wps, dict):
        wps = {}
        omrat.waypoints = wps
    fresh = rebuild_waypoints(omrat.segment_data, wps)
    wps.clear()
    wps.update(fresh)
    return wps


def _registry_before(omrat: "OMRAT") -> dict[str, Junction]:
    handler = getattr(omrat, 'junctions', None)
    if handler is None:
        return {}
    handler.rebuild_from_segments(omrat.segment_data, prefer_user=True)
    return copy.deepcopy(handler.registry)


def _after_edit(omrat: "OMRAT", before: dict[str, Junction], leg_id: str) -> dict[str, list[str]]:
    """Map, route table and junction registry after one leg end moved."""
    sd = omrat.segment_data
    geoms = getattr(omrat, 'qgis_geoms', None)
    if geoms is not None:
        # traffic_data is keyed by the direction names: a leg whose
        # heading changed keeps them (the canvas rewrite may re-derive them).
        dirs = copy.deepcopy(sd[leg_id].get('Dirs'))
        prev = getattr(geoms, '_propagating_vertex_move', False)
        geoms._propagating_vertex_move = True
        try:
            geoms._refresh_leg_views(int(leg_id))
        except (TypeError, ValueError):
            pass
        finally:
            geoms._propagating_vertex_move = prev
        if dirs is not None:
            sd[leg_id]['Dirs'] = dirs
            if isinstance(getattr(geoms, 'leg_dirs', None), dict):
                geoms.leg_dirs[leg_id] = dirs
    result: dict[str, list[str]] = {'trimmed': [], 'extended': []}
    handler = getattr(omrat, 'junctions', None)
    if handler is not None:
        handler.rebuild_from_segments(sd, prefer_user=True)
        result = carry_user_matrices(before, handler.registry, sd)
    if geoms is not None:
        # The passage line moved: AIS junction counts on the leg are stale.
        geoms._invalidate_junction_ais({leg_id})
    matrix_dlg = getattr(omrat, '_junction_matrix_dialog', None)
    if matrix_dlg is not None:
        try:
            if matrix_dlg.isVisible():
                matrix_dlg.reload()
        except RuntimeError:
            pass
    iface = getattr(omrat, 'iface', None)
    canvas = iface.mapCanvas() if iface is not None else None
    if canvas is not None:
        canvas.refresh()
    return result


def apply_detach(omrat: "OMRAT", wp_id: str, leg_id: str, side: str, gap_m: float) -> dict[str, Any]:
    """Detach ``leg_id``'s ``side`` end from ``wp_id`` (see
    :func:`geometries.waypoints.detach_leg_end`) and refresh the plugin.
    Returns ``{'new_wp', 'trimmed', 'extended'}``; raises
    :class:`WaypointEditError` and changes nothing when it cannot."""
    leg_id = str(leg_id)
    wps = sync_waypoints(omrat)
    before = _registry_before(omrat)
    new_wp = detach_leg_end(wps, omrat.segment_data, str(wp_id), leg_id, side, gap_m)
    return {'new_wp': new_wp, **_after_edit(omrat, before, leg_id)}


def apply_attach(omrat: "OMRAT", leg_id: str, side: str, wp_id: str) -> dict[str, Any]:
    """Connect ``leg_id``'s ``side`` end to ``wp_id`` (see
    :func:`geometries.waypoints.attach_leg_end`) and refresh the plugin.
    Returns ``{'old_wp', 'trimmed', 'extended'}``."""
    leg_id = str(leg_id)
    wps = sync_waypoints(omrat)
    before = _registry_before(omrat)
    old_wp = attach_leg_end(wps, omrat.segment_data, leg_id, side, str(wp_id))
    return {'old_wp': old_wp, **_after_edit(omrat, before, leg_id)}


def new_crossings_after_attach(omrat: "OMRAT", leg_id: str, side: str, wp_id: str) -> list[str]:
    """Legs that ``leg_id`` would newly cross once connected (dry run)."""
    sd = copy.deepcopy(omrat.segment_data)
    wps = copy.deepcopy(getattr(omrat, 'waypoints', None) or {})
    before = set(crossings_of_leg(sd, leg_id))
    attach_leg_end(wps, sd, str(leg_id), side, str(wp_id))
    return sorted(set(crossings_of_leg(sd, leg_id)) - before)


def _distance_text(metres: float) -> str:
    return f"{metres / 1000:.1f} km" if metres >= 1000 else f"{metres:.0f} m"


def _leg_text(leg_id: str, segment_data: dict[str, Any]) -> str:
    return f"{leg_name(leg_id, segment_data)}  (id {leg_id})"


def _matrix_note(result: dict[str, Any]) -> str:
    notes = []
    if result.get('trimmed'):
        notes.append("hand-edited junction matrix kept (rows renormalised)")
    if result.get('extended'):
        notes.append("hand-edited junction matrix got the new leg at 0 % -- check it")
    return ("; " + "; ".join(notes)) if notes else ""


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class WaypointEditorDialog(QDialog):
    """Modeless editor: waypoints on the left, the selected one's legs right."""

    LAYER_NAME = "Waypoints (editor)"

    def __init__(self, omrat: "OMRAT", parent=None):
        super().__init__(parent)
        self.omrat = omrat
        self.setWindowTitle(self._tr("Waypoint editor"))
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(780, 520)
        self._bands: list[QgsRubberBand] = []
        self._preview: QgsRubberBand | None = None
        self._layer: QgsVectorLayer | None = None
        self._pick_tool = None
        self._prev_tool = None
        self._sig: tuple = ()
        self._zoom_on_select = True

        layout = QVBoxLayout(self)
        intro = QLabel(self._tr(
            "Pick a waypoint in the list or on the map.  Detach moves a leg end back along "
            "its own leg onto a new waypoint, so it cannot cross the legs that stay.  Connect "
            "moves a leg end onto the selected waypoint.  Changes cannot be undone."
        ))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        body = QHBoxLayout()
        layout.addLayout(body, 1)

        left = QVBoxLayout()
        left.addWidget(QLabel(self._tr("Waypoints")))
        self.lst_wp = QListWidget()
        left.addWidget(self.lst_wp, 1)
        self.btn_pick = QPushButton(self._tr("Pick on map"))
        left.addWidget(self.btn_pick)
        body.addLayout(left, 2)

        right = QVBoxLayout()
        self.lbl_wp = QLabel("")
        self.lbl_wp.setWordWrap(True)
        right.addWidget(self.lbl_wp)

        grp_on = QGroupBox(self._tr("Legs on this waypoint"))
        on_lay = QVBoxLayout(grp_on)
        self.lst_legs = QListWidget()
        self.lst_legs.setMinimumHeight(110)     # four rows: a typical junction without scrolling
        on_lay.addWidget(self.lst_legs, 1)
        gap_row = QHBoxLayout()
        gap_row.addWidget(QLabel(self._tr("Move the end back along the leg by")))
        self.sp_gap = QDoubleSpinBox()
        self.sp_gap.setDecimals(0)
        self.sp_gap.setSuffix(" m")
        self.sp_gap.setSingleStep(50)
        gap_row.addWidget(self.sp_gap)
        on_lay.addLayout(gap_row)
        self.lbl_gap = QLabel("")
        self.lbl_gap.setWordWrap(True)
        on_lay.addWidget(self.lbl_gap)
        self.btn_detach = QPushButton(self._tr("Detach selected leg"))
        on_lay.addWidget(self.btn_detach)
        right.addWidget(grp_on, 1)

        grp_add = QGroupBox(self._tr("Connect a leg end to this waypoint"))
        add_lay = QVBoxLayout(grp_add)
        self.cmb_cand = QComboBox()
        add_lay.addWidget(self.cmb_cand)
        self.btn_attach = QPushButton(self._tr("Connect"))
        add_lay.addWidget(self.btn_attach)
        right.addWidget(grp_add)

        self.btn_matrix = QPushButton(self._tr("Open junction matrix..."))
        right.addWidget(self.btn_matrix)
        body.addLayout(right, 3)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.lst_wp.currentRowChanged.connect(self._on_wp_changed)
        self.lst_legs.currentRowChanged.connect(self._on_leg_changed)
        self.sp_gap.valueChanged.connect(self._update_preview)
        self.cmb_cand.currentIndexChanged.connect(self._on_candidate_changed)
        self.btn_detach.clicked.connect(self._on_detach)
        self.btn_attach.clicked.connect(self._on_attach)
        self.btn_matrix.clicked.connect(self._on_open_matrix)
        self.btn_pick.clicked.connect(self._start_pick)

        self.reload()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _tr(self, text: str) -> str:
        tr = getattr(self.omrat, 'tr', None)
        return tr(text) if callable(tr) else text

    def _segments(self) -> dict[str, Any]:
        return getattr(self.omrat, 'segment_data', None) or {}

    def _signature(self) -> tuple:
        return tuple(
            (str(k), v.get('Start_Point'), v.get('End_Point'))
            for k, v in self._segments().items() if isinstance(v, dict)
        )

    def current_wp(self) -> str | None:
        item = self.lst_wp.currentItem()
        return None if item is None else str(item.data(Qt.ItemDataRole.UserRole))

    def current_leg_end(self) -> tuple[str, str] | None:
        item = self.lst_legs.currentItem()
        return None if item is None else tuple(item.data(Qt.ItemDataRole.UserRole))

    def current_candidate(self) -> tuple[str, str] | None:
        data = self.cmb_cand.currentData()
        return None if data is None else tuple(data)

    def _junction(self, wp_id: str) -> Junction | None:
        handler = getattr(self.omrat, 'junctions', None)
        xy = (getattr(self.omrat, 'waypoints', None) or {}).get(wp_id)
        if handler is None or xy is None:
            return None
        return handler.registry.get(junction_id_for_point(xy))

    def reload(self, select: str | None = None) -> None:
        """Re-read waypoints (legs may have changed since the last look)."""
        keep = select if select is not None else self.current_wp()
        wps = sync_waypoints(self.omrat)
        handler = getattr(self.omrat, 'junctions', None)
        if handler is not None:
            handler.rebuild_from_segments(self.omrat.segment_data, prefer_user=True)
        sd = self._segments()
        self._sig = self._signature()
        self.lst_wp.blockSignals(True)
        self.lst_wp.clear()
        row_to_select = 0
        for i, wid in enumerate(sorted(wps, key=lambda k: (len(str(k)), str(k)))):
            n = len(incident_legs(sd, wid))
            j = self._junction(wid)
            kind = f"junction, {j.source} matrix" if j is not None and n >= 2 else "leg end"
            item = QListWidgetItem(f"WP {wid}    {n} leg{'s' if n != 1 else ''}    ({kind})")
            item.setData(Qt.ItemDataRole.UserRole, str(wid))
            self.lst_wp.addItem(item)
            if str(wid) == str(keep):
                row_to_select = i
        self.lst_wp.blockSignals(False)
        self._refresh_layer()
        # A reload never moves the map; only picking a row in the list does.
        self._zoom_on_select = False
        if self.lst_wp.count():
            self.lst_wp.setCurrentRow(row_to_select)
            self._on_wp_changed(row_to_select)
        else:
            self._on_wp_changed(-1)
        self._zoom_on_select = True

    def select_waypoint(self, wp_id: str, *, zoom: bool = True) -> bool:
        for row in range(self.lst_wp.count()):
            if str(self.lst_wp.item(row).data(Qt.ItemDataRole.UserRole)) == str(wp_id):
                self._zoom_on_select = zoom
                if self.lst_wp.currentRow() == row:
                    self._on_wp_changed(row)
                else:
                    self.lst_wp.setCurrentRow(row)
                self._zoom_on_select = True
                return True
        return False

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_wp_changed(self, _row: int) -> None:
        wid = self.current_wp()
        sd = self._segments()
        wps = getattr(self.omrat, 'waypoints', None) or {}
        self.lst_legs.blockSignals(True)
        self.lst_legs.clear()
        self.cmb_cand.blockSignals(True)
        self.cmb_cand.clear()
        if wid is None or wid not in wps:
            self.lbl_wp.setText(self._tr("No waypoints: draw or load a route first."))
        else:
            legs = incident_legs(sd, wid)
            j = self._junction(wid)
            lon, lat = wps[wid]
            src = f", junction matrix: {j.source}" if j is not None and len(legs) >= 2 else ""
            self.lbl_wp.setText(f"WP {wid} at {lon:.6f} {lat:.6f} -- {len(legs)} leg(s){src}")
            for leg_id, side in legs:
                item = QListWidgetItem(f"{_leg_text(leg_id, sd)}  -  {side}")
                item.setData(Qt.ItemDataRole.UserRole, [leg_id, side])
                self.lst_legs.addItem(item)
            for leg_id, side, dist in attach_candidates(wps, sd, wid):
                self.cmb_cand.addItem(
                    f"{_leg_text(leg_id, sd)}  -  {side},  {_distance_text(dist)} away", [leg_id, side],
                )
        self.lst_legs.blockSignals(False)
        self.cmb_cand.blockSignals(False)
        if self.lst_legs.count():
            self.lst_legs.setCurrentRow(0)
        self._on_leg_changed(self.lst_legs.currentRow())
        self.btn_attach.setEnabled(self.cmb_cand.count() > 0)
        j = self._junction(wid) if wid is not None else None
        self.btn_matrix.setEnabled(j is not None)
        self._highlight()
        if self._zoom_on_select and wid is not None:
            self._zoom_to(wid)

    def _on_leg_changed(self, _row: int) -> None:
        wid = self.current_wp()
        end = self.current_leg_end()
        sd = self._segments()
        ok = False
        if wid is not None and end is not None and len(incident_legs(sd, wid)) >= 2:
            min_m, suggested, max_m = detach_limits(
                getattr(self.omrat, 'waypoints', None) or {}, sd, wid, end[0], end[1],
            )
            lo, hi = math.ceil(min_m) + 1, math.floor(max_m)
            if lo <= hi:
                ok = True
                self.sp_gap.blockSignals(True)
                self.sp_gap.setRange(lo, hi)
                self.sp_gap.setValue(min(max(round(suggested / 10) * 10, lo), hi))
                self.sp_gap.blockSignals(False)
                self.lbl_gap.setText(self._tr(
                    "At least {lo} m: closer, and Update all distributions offers to join the "
                    "two waypoints again.  At most {hi} m (half the leg)."
                ).format(lo=lo, hi=hi))
            else:
                self.lbl_gap.setText(self._tr("This leg is too short to detach."))
        elif end is not None:
            self.lbl_gap.setText(self._tr("The only leg on this waypoint: nothing to detach from."))
        else:
            self.lbl_gap.setText("")
        self.sp_gap.setEnabled(ok)
        self.btn_detach.setEnabled(ok)
        self._update_preview()

    def _on_candidate_changed(self, _index: int) -> None:
        self._highlight()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(
            self, title, text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _stale(self) -> bool:
        """Legs changed behind the dialog's back: reload instead of acting."""
        if self._signature() == self._sig:
            return False
        self.reload()
        QMessageBox.information(
            self, self._tr("Waypoint editor"),
            self._tr("The legs changed since the list was drawn; it has been refreshed. Pick again."),
        )
        return True

    def _on_detach(self) -> None:
        wid, end = self.current_wp(), self.current_leg_end()
        if wid is None or end is None or self._stale():
            return
        sd = self._segments()
        leg_id, side = end
        gap = float(self.sp_gap.value())
        text = self._tr(
            "Detach the {side} of {leg} from WP {wp}?\n\n"
            "The end moves {gap} back along the leg onto a new waypoint.  The other legs "
            "on WP {wp} stay where they are.  If the leg has AIS traffic, run Update AIS on it "
            "afterwards: its passage line moves with the leg.\n\nThis cannot be undone."
        ).format(side=side, leg=_leg_text(leg_id, sd), wp=wid, gap=_distance_text(gap))
        if not self._confirm(self._tr("Detach leg"), text):
            return
        try:
            result = apply_detach(self.omrat, wid, leg_id, side, gap)
        except WaypointEditError as exc:
            QMessageBox.warning(self, self._tr("Detach leg"), str(exc))
            return
        self.reload(select=wid)
        self._notify(
            f"Detached {_leg_text(leg_id, sd)} from WP {wid}; its {side} is now WP {result['new_wp']}"
            f"{_matrix_note(result)}."
        )

    def _on_attach(self) -> None:
        wid, cand = self.current_wp(), self.current_candidate()
        if wid is None or cand is None or self._stale():
            return
        sd = self._segments()
        wps = getattr(self.omrat, 'waypoints', None) or {}
        leg_id, side = cand
        dist = next((d for lid, s, d in attach_candidates(wps, sd, wid) if (lid, s) == (leg_id, side)), 0.0)
        text = self._tr(
            "Connect the {side} of {leg} to WP {wp}?\n\n"
            "The end moves {dist} onto the waypoint; legs on its current waypoint stay where "
            "they are.  If the leg has AIS traffic, run Update AIS on it afterwards."
        ).format(side=side, leg=_leg_text(leg_id, sd), wp=wid, dist=_distance_text(dist))
        try:
            crossed = new_crossings_after_attach(self.omrat, leg_id, side, wid)
        except WaypointEditError as exc:
            QMessageBox.warning(self, self._tr("Connect leg"), str(exc))
            return
        if crossed:
            text += self._tr(
                "\n\nAfter the move it crosses {legs}; Update all distributions will offer to split them."
            ).format(legs=", ".join(leg_name(c, sd) for c in crossed))
        text += self._tr("\n\nThis cannot be undone.")
        if not self._confirm(self._tr("Connect leg"), text):
            return
        try:
            result = apply_attach(self.omrat, leg_id, side, wid)
        except WaypointEditError as exc:
            QMessageBox.warning(self, self._tr("Connect leg"), str(exc))
            return
        self.reload(select=wid)
        self._notify(
            f"Connected the {side} of {_leg_text(leg_id, sd)} to WP {wid}{_matrix_note(result)}."
        )

    def _on_open_matrix(self) -> None:
        wid = self.current_wp()
        j = self._junction(wid) if wid is not None else None
        if j is None:
            return
        from omrat_utils.junction_matrix_dialog import open_junction_dialog
        open_junction_dialog(self.omrat)
        dlg = getattr(self.omrat, '_junction_matrix_dialog', None)
        if dlg is not None:
            dlg.select_junction(j.junction_id)

    def _notify(self, message: str) -> None:
        notifier = getattr(self.omrat, 'notifier', None)
        if notifier is None:
            return
        try:
            notifier.display_message(message, duration=10)
        except Exception:  # nosec B110 B112
            pass

    # ------------------------------------------------------------------
    # Map: labelled waypoint layer, highlight, preview, pick tool
    # ------------------------------------------------------------------

    def _canvas(self):
        iface = getattr(self.omrat, 'iface', None)
        return iface.mapCanvas() if iface is not None else None

    def _ensure_layer(self) -> QgsVectorLayer | None:
        layer = self._layer
        try:
            if layer is not None and QgsProject.instance().mapLayer(layer.id()) is not None:
                return layer
        except RuntimeError:
            pass
        layer = QgsVectorLayer("Point?crs=EPSG:4326", self.LAYER_NAME, "memory")
        fields = QgsFields()
        fields.append(QgsField('wp', QMetaType.Type.QString))
        fields.append(QgsField('legs', QMetaType.Type.Int))
        layer.dataProvider().addAttributes(fields.toList())
        layer.updateFields()
        symbol = QgsMarkerSymbol.createSimple(
            {'name': 'circle', 'color': '255,255,255', 'outline_color': '40,40,40', 'size': '2.2'})
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        settings = QgsPalLayerSettings()
        settings.fieldName = "'WP ' || \"wp\""
        settings.isExpression = True
        settings.enabled = True
        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
        layer.setLabelsEnabled(True)
        QgsProject.instance().addMapLayer(layer)
        self._layer = layer
        return layer

    def _refresh_layer(self) -> None:
        layer = self._ensure_layer()
        if layer is None:
            return
        sd = self._segments()
        provider = layer.dataProvider()
        provider.truncate()
        feats = []
        for wid, (x, y) in (getattr(self.omrat, 'waypoints', None) or {}).items():
            feat = QgsFeature(layer.fields())
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
            feat.setAttributes([str(wid), len(incident_legs(sd, wid))])
            feats.append(feat)
        provider.addFeatures(feats)
        layer.updateExtents()
        layer.triggerRepaint()

    def _remove_layer(self) -> None:
        layer, self._layer = self._layer, None
        if layer is None:
            return
        try:
            if QgsProject.instance().mapLayer(layer.id()) is not None:
                QgsProject.instance().removeMapLayer(layer.id())
        except RuntimeError:
            pass

    def _band(self, geom_type) -> QgsRubberBand | None:
        canvas = self._canvas()
        if canvas is None:
            return None
        band = QgsRubberBand(canvas, geom_type)
        self._bands.append(band)
        return band

    def _clear_bands(self) -> None:
        canvas = self._canvas()
        for band in self._bands:
            try:
                band.reset(QgsWkbTypes.GeometryType.LineGeometry)
                if canvas is not None:
                    canvas.scene().removeItem(band)
            except Exception:  # nosec B110 B112
                pass
        self._bands = []
        self._preview = None

    def highlighted(self) -> int:
        """Number of highlight bands on the canvas (for tests)."""
        return len(self._bands)

    def _leg_geom(self, leg_id: str) -> QgsGeometry | None:
        seg = self._segments().get(str(leg_id))
        if not isinstance(seg, dict):
            return None
        return QgsGeometry.fromWkt(f"LINESTRING({seg['Start_Point']}, {seg['End_Point']})")

    def _highlight(self) -> None:
        """The waypoint (marker), its legs (yellow) and the picked
        candidate (orange)."""
        self._clear_bands()
        wid = self.current_wp()
        wps = getattr(self.omrat, 'waypoints', None) or {}
        if wid is None or wid not in wps or self._canvas() is None:
            return
        crs = QgsCoordinateReferenceSystem("EPSG:4326")
        legs = [(lid, QColor(255, 220, 0, 200)) for lid, _s in incident_legs(self._segments(), wid)]
        cand = self.current_candidate()
        if cand is not None:
            legs.append((cand[0], QColor(255, 120, 0, 170)))
        for lid, colour in legs:
            geom = self._leg_geom(lid)
            band = self._band(QgsWkbTypes.GeometryType.LineGeometry)
            if geom is None or band is None:
                continue
            band.setColor(colour)
            band.setWidth(8)
            band.setToGeometry(geom, crs)
        marker = self._band(QgsWkbTypes.GeometryType.PointGeometry)
        if marker is not None:
            marker.setColor(QColor(220, 0, 0, 220))
            marker.setIcon(QgsRubberBand.IconType.ICON_CIRCLE)
            marker.setIconSize(14)
            marker.setWidth(3)
            marker.setToGeometry(QgsGeometry.fromPointXY(QgsPointXY(*wps[wid])), crs)
        self._update_preview()

    def _update_preview(self, *_args) -> None:
        """Where a detached end would go (blue cross)."""
        canvas = self._canvas()
        if self._preview is not None:
            try:
                self._preview.reset(QgsWkbTypes.GeometryType.PointGeometry)
            except Exception:  # nosec B110 B112
                pass
        end = self.current_leg_end()
        if canvas is None or end is None or not self.btn_detach.isEnabled():
            return
        try:
            xy = detach_point(self._segments(), end[0], end[1], float(self.sp_gap.value()))
        except WaypointEditError:
            return
        if self._preview is None:
            self._preview = self._band(QgsWkbTypes.GeometryType.PointGeometry)
            if self._preview is None:
                return
            self._preview.setColor(QColor(0, 90, 220, 230))
            self._preview.setIcon(QgsRubberBand.IconType.ICON_X)
            self._preview.setIconSize(14)
            self._preview.setWidth(3)
        self._preview.setToGeometry(
            QgsGeometry.fromPointXY(QgsPointXY(*xy)), QgsCoordinateReferenceSystem("EPSG:4326"))

    def _zoom_to(self, wid: str) -> None:
        canvas = self._canvas()
        wps = getattr(self.omrat, 'waypoints', None) or {}
        if canvas is None or wid not in wps:
            return
        from geometries.route_validation import leg_length_m
        sd = self._segments()
        lengths = [leg_length_m(sd[lid]) for lid, _s in incident_legs(sd, wid)]
        radius = max(0.35 * min([v for v in lengths if v > 0], default=2000.0), 300.0)
        lon, lat = wps[wid]
        dlat = radius / 111_320.0
        dlon = radius / (111_320.0 * max(math.cos(math.radians(lat)), 0.01))
        rect = QgsRectangle(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
        canvas_crs = canvas.mapSettings().destinationCrs()
        src = QgsCoordinateReferenceSystem("EPSG:4326")
        if canvas_crs.authid() != src.authid():
            try:
                rect = QgsCoordinateTransform(src, canvas_crs, QgsProject.instance()).transformBoundingBox(rect)
            except Exception:  # nosec B110
                pass
        canvas.setExtent(rect)
        canvas.refresh()

    def _start_pick(self) -> None:
        canvas = self._canvas()
        if canvas is None:
            return
        from omrat_utils import PointTool
        self._prev_tool = canvas.mapTool()
        self._pick_tool = PointTool(canvas)
        self._pick_tool.canvasClicked.connect(self._on_map_pick)
        canvas.setMapTool(self._pick_tool)
        self._notify(self._tr("Click a waypoint on the map."))

    def _end_pick(self) -> None:
        canvas = self._canvas()
        tool, self._pick_tool = self._pick_tool, None
        if canvas is None or tool is None:
            return
        try:
            if canvas.mapTool() is tool:
                if self._prev_tool is not None:
                    canvas.setMapTool(self._prev_tool)
                else:
                    canvas.unsetMapTool(tool)
        except RuntimeError:
            pass
        self._prev_tool = None

    def _on_map_pick(self, point) -> None:
        canvas = self._canvas()
        xy = QgsPointXY(point.x(), point.y())
        if canvas is not None:
            canvas_crs = canvas.mapSettings().destinationCrs()
            src = QgsCoordinateReferenceSystem("EPSG:4326")
            if canvas_crs.authid() != src.authid():
                try:
                    xy = QgsCoordinateTransform(canvas_crs, src, QgsProject.instance()).transform(xy)
                except Exception:  # nosec B110
                    pass
        self.pick_at((xy.x(), xy.y()))
        self._end_pick()

    def pick_at(self, xy: tuple[float, float]) -> str | None:
        """Select the waypoint under ``xy`` (EPSG:4326), if any."""
        geoms = getattr(self.omrat, 'qgis_geoms', None)
        if geoms is None:
            return None
        _snapped, wid = geoms.snap_to_waypoint(xy)
        if wid is None:
            self._notify(self._tr("No waypoint there. Click closer to a leg end."))
            return None
        self.select_waypoint(wid, zoom=False)
        self.raise_()
        return wid

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if self._signature() != self._sig:
                self.reload()

    def reject(self) -> None:
        self.close()

    def closeEvent(self, event) -> None:
        self._cleanup()
        super().closeEvent(event)

    def _cleanup(self) -> None:
        self._end_pick()
        self._clear_bands()
        self._remove_layer()
        if getattr(self.omrat, '_waypoint_editor_dlg', None) is self:
            self.omrat._waypoint_editor_dlg = None


def open_waypoint_editor(omrat: "OMRAT") -> WaypointEditorDialog | None:
    """Menu entry: open the editor (modeless) or raise the open one."""
    existing = getattr(omrat, '_waypoint_editor_dlg', None)
    if existing is not None:
        try:
            existing.reload()
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return existing
        except RuntimeError:
            omrat._waypoint_editor_dlg = None
    if not (getattr(omrat, 'segment_data', None) or {}):
        QMessageBox.information(
            getattr(omrat, 'main_widget', None), "Waypoint editor",
            "There are no legs yet. Draw or load a route first.",
        )
        return None
    dlg = WaypointEditorDialog(omrat, getattr(omrat, 'main_widget', None))
    # Keep a Python reference, otherwise the modeless dialog is collected.
    omrat._waypoint_editor_dlg = dlg
    dlg.show()
    return dlg
