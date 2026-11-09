"""**Settings > Speed restriction zones...** -- the project's speed limits.

Modeless (the canvas stays usable for drawing): the zones on top, the
rules of the selected zone below.  Every edit is written straight to
``omrat.speed_zones`` (:class:`omrat_utils.speed_restriction_zones.SpeedRestrictionZones`),
which redraws the map layer and reports overlaps; this dialog only shows
and collects.  Rules are tried from the top and the first one that
matches a ship wins (see :mod:`compute.speed_restriction`).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from qgis.PyQt.QtCore import QSettings, Qt, QTimer
from qgis.PyQt.QtGui import QBrush, QColor
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.gui import QgsRubberBand

from compute.speed_restriction import (
    DEFAULT_LIMIT_KN, length_boundaries, length_class_bounds, new_rule, zone_label,
)
from omrat_utils.speed_restriction_zones import guess_field
from omrat_utils.table_clipboard import format_number, parse_number

if TYPE_CHECKING:
    from omrat import OMRAT

LAST_DIR_KEY = 'omrat/speed_zones_dir'

ZONE_COLS = ('On', 'Id', 'Name', 'Limit (kn)', 'Compliance (%)', 'Applies to', 'Legs')
COL_ON, COL_ID, COL_NAME, COL_LIMIT, COL_COMPLIANCE, COL_APPLIES, COL_LEGS = range(len(ZONE_COLS))
RULE_COLS = ('Ship type', 'Length from (m)', 'Length to (m)', 'Restricted', 'Limit (kn)', 'Compliance (%)')
RCOL_TYPE, RCOL_FROM, RCOL_TO, RCOL_RESTRICTED, RCOL_LIMIT, RCOL_COMPLIANCE = range(len(RULE_COLS))
#: Zone cells a single click opens for editing (like Width / Tangent % in the route table).
SINGLE_CLICK_EDIT = (COL_NAME, COL_LIMIT, COL_COMPLIANCE)

_PROBLEM_BG = QColor(255, 205, 205)
_CHECKED = Qt.CheckState.Checked
_UNCHECKED = Qt.CheckState.Unchecked
_EDITABLE = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
_READ_ONLY = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
_CHECKABLE = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable


def _num_text(value: float | None) -> str:
    return '' if value is None else format_number(float(value))


def _number(text: str) -> tuple[bool, float | None]:
    """``(valid, value)`` of a cell; a blank cell is valid with ``None``."""
    try:
        return True, parse_number(text)
    except ValueError:
        return False, None


def legs_tooltip(effects: list[dict[str, Any]]) -> str:
    """One line per leg: share inside, mean speed per direction before -> along the leg."""
    lines = []
    for leg in effects:
        share = 100.0 * sum((leg.get('shares') or {}).values())
        where = f'{share:.0f} % inside' if share > 0 else 'waypoint only'
        speeds = []
        for d in leg.get('dirs') or []:
            before, after = d.get('before'), d.get('leg')
            if before is None or after is None:
                continue
            speeds.append(f"dir {d['dir'] + 1} {before:.1f} -> {after:.1f} kn")
        lines.append(f"{leg['name']}: {where}" + (f", {', '.join(speeds)}" if speeds else ''))
    return '\n'.join(lines) if lines else 'No leg passes through this zone.'


class SpeedRestrictionDialog(QDialog):
    """Modeless editor of the project's speed restriction zones."""

    def __init__(self, omrat: "OMRAT", parent=None) -> None:
        super().__init__(parent)
        self.omrat = omrat
        self.zones_handler = omrat.speed_zones
        self.setWindowTitle(self._tr('Speed restriction zones'))
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(860, 620)
        self._filling = False
        self._reload_pending = False
        self._bands: list[QgsRubberBand] = []
        self._selected_id: int | None = None

        layout = QVBoxLayout(self)
        intro = QLabel(self._tr(
            "Ships sail at most the zone's limit inside the zone.  Along a leg the model uses the "
            "time-equivalent speed (share of the leg inside the zone); crossing, merging, bend and "
            "powered Cat II use the speed at the waypoint.  Only the accident frequencies change: "
            "a lower speed means more ships in the area at the same time, so most frequencies go up; "
            "overtaking and powered Cat II go down.  Zones may not overlap."))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.tw_zones = QTableWidget(0, len(ZONE_COLS))
        self.tw_zones.setHorizontalHeaderLabels([self._tr(c) for c in ZONE_COLS])
        self.tw_zones.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tw_zones.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tw_zones.verticalHeader().setVisible(False)
        header = self.tw_zones.horizontalHeader()
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tw_zones, 2)

        buttons = QHBoxLayout()
        self.btn_draw = QPushButton(self._tr('Draw new'))
        self.btn_load = QPushButton(self._tr('Load from file...'))
        self.btn_remove = QPushButton(self._tr('Remove'))
        self.btn_zoom = QPushButton(self._tr('Zoom to'))
        for btn in (self.btn_draw, self.btn_load, self.btn_remove, self.btn_zoom):
            buttons.addWidget(btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.lbl_problems = QLabel('')
        self.lbl_problems.setWordWrap(True)
        self.lbl_problems.setStyleSheet('color: #b00000;')
        layout.addWidget(self.lbl_problems)

        self.grp_rules = QGroupBox(self._tr('Rules for the selected zone'))
        rules_layout = QVBoxLayout(self.grp_rules)
        hint = QLabel(self._tr(
            "Without rules the zone applies to every ship.  The first rule from the top that matches a "
            "ship wins; ships no rule matches get the zone's limit.  A blank limit / compliance cell "
            "means the zone's value.  Lengths are the boundaries of the length classes."))
        hint.setWordWrap(True)
        rules_layout.addWidget(hint)
        self.tw_rules = QTableWidget(0, len(RULE_COLS))
        self.tw_rules.setHorizontalHeaderLabels([self._tr(c) for c in RULE_COLS])
        self.tw_rules.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tw_rules.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tw_rules.horizontalHeader().setSectionResizeMode(RCOL_TYPE, QHeaderView.ResizeMode.Stretch)
        rules_layout.addWidget(self.tw_rules, 1)
        rule_buttons = QHBoxLayout()
        self.btn_rule_add = QPushButton(self._tr('Add rule'))
        self.btn_rule_remove = QPushButton(self._tr('Remove rule'))
        self.btn_rule_up = QPushButton(self._tr('Up'))
        self.btn_rule_down = QPushButton(self._tr('Down'))
        for btn in (self.btn_rule_add, self.btn_rule_remove, self.btn_rule_up, self.btn_rule_down):
            rule_buttons.addWidget(btn)
        rule_buttons.addStretch(1)
        rules_layout.addLayout(rule_buttons)
        layout.addWidget(self.grp_rules, 2)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.close)
        layout.addWidget(box)

        self.tw_zones.itemChanged.connect(self._on_zone_item_changed)
        self.tw_zones.itemSelectionChanged.connect(self._on_zone_selected)
        self.tw_zones.cellClicked.connect(self._on_zone_cell_clicked)
        self.tw_rules.itemChanged.connect(self._on_rule_item_changed)
        self.btn_draw.clicked.connect(self._on_draw)
        self.btn_load.clicked.connect(self._on_load)
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_zoom.clicked.connect(lambda _checked=False: self.zoom_to_selected())
        self.btn_rule_add.clicked.connect(lambda _checked=False: self.add_rule())
        self.btn_rule_remove.clicked.connect(lambda _checked=False: self.remove_rule())
        self.btn_rule_up.clicked.connect(lambda _checked=False: self.move_rule(-1))
        self.btn_rule_down.clicked.connect(lambda _checked=False: self.move_rule(1))

        self.zones_handler.add_listener(self._on_zones_changed)
        zones = self.zones_handler.zones
        self._selected_id = zones[0]['id'] if zones else None
        self.reload()

    @staticmethod
    def _tr(text: str) -> str:
        return text

    def _testing(self) -> bool:
        return bool(getattr(self.omrat, 'testing', False))

    # ------------------------------------------------------------------
    # Filling the tables
    # ------------------------------------------------------------------

    def _on_zones_changed(self) -> None:
        """Handler listener: reload after the change has settled (the
        change may come from inside one of this dialog's own signals)."""
        if self._reload_pending:
            return
        self._reload_pending = True
        QTimer.singleShot(0, self._deferred_reload)

    def _deferred_reload(self) -> None:
        self._reload_pending = False
        try:
            self.reload()
        except RuntimeError:
            pass   # dialog already deleted

    def reload(self) -> None:
        """Refill both tables from the handler, keeping the selected zone."""
        zones = self.zones_handler.zones
        ids = [z['id'] for z in zones]
        if self._selected_id not in ids:
            self._selected_id = ids[0] if ids else None
        problems = self.zones_handler.problems()
        self._filling = True
        try:
            self.tw_zones.setRowCount(len(zones))
            for row, zone in enumerate(zones):
                self._fill_zone_row(row, zone, problems.get(zone['id'], []))
            self.tw_zones.resizeColumnsToContents()
            self.tw_zones.horizontalHeader().setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
            if self._selected_id is not None:
                self.tw_zones.selectRow(ids.index(self._selected_id))
            else:
                self.tw_zones.clearSelection()
        finally:
            self._filling = False
        self._fill_rules()
        self._show_problems(problems)
        self._update_buttons()
        self._highlight()

    def _fill_zone_row(self, row: int, zone: dict[str, Any], problems: list[str]) -> None:
        on = QTableWidgetItem('')
        on.setFlags(_CHECKABLE)
        on.setCheckState(_CHECKED if zone.get('enabled', True) else _UNCHECKED)
        cells = {
            COL_ON: on,
            COL_ID: self._item(str(zone['id']), editable=False),
            COL_NAME: self._item(str(zone.get('name') or ''), editable=True),
            COL_LIMIT: self._item(_num_text(zone.get('limit_kn')), editable=True),
            COL_COMPLIANCE: self._item(_num_text(zone.get('compliance_pct')), editable=True),
        }
        rules = zone.get('rules') or []
        cells[COL_APPLIES] = self._item(
            'All ships' if not rules else f"{len(rules)} rule{'s' if len(rules) != 1 else ''}", editable=False)
        effects = self.zones_handler.leg_effects(zone)
        legs = self._item(str(len(effects)), editable=False)
        legs.setToolTip(legs_tooltip(effects))
        cells[COL_LEGS] = legs
        for col, item in cells.items():
            if problems:
                item.setBackground(QBrush(_PROBLEM_BG))
                if col != COL_LEGS:
                    item.setToolTip('\n'.join(problems))
            self.tw_zones.setItem(row, col, item)

    @staticmethod
    def _item(text: str, editable: bool) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(_EDITABLE if editable else _READ_ONLY)
        return item

    def _show_problems(self, problems: dict[int, list[str]]) -> None:
        enabled = {z['id'] for z in self.zones_handler.zones if z.get('enabled', True)}
        blocking = any(zid in enabled for zid in problems)
        own = problems.get(self._selected_id, []) if self._selected_id is not None else []
        lines = list(dict.fromkeys(own))
        if blocking:
            lines.append(self._tr('Run model is blocked until the red zones are fixed or switched off.'))
        self.lbl_problems.setText('\n'.join(lines))

    def _update_buttons(self) -> None:
        has = self._selected_id is not None
        self.btn_remove.setEnabled(has)
        self.btn_zoom.setEnabled(has)
        self.grp_rules.setEnabled(has)
        drawing = self.zones_handler.is_drawing()
        self.btn_draw.setText(self._tr('Cancel drawing') if drawing else self._tr('Draw new'))
        zone = self.selected_zone()
        self.grp_rules.setTitle(
            self._tr('Rules for {zone}').format(zone=zone_label(zone)) if zone else self._tr('Rules'))

    # ------------------------------------------------------------------
    # Zones table
    # ------------------------------------------------------------------

    def selected_zone(self) -> dict[str, Any] | None:
        return self.zones_handler.zone(self._selected_id) if self._selected_id is not None else None

    def select_zone(self, zone_id: int) -> None:
        self._selected_id = int(zone_id)
        self.reload()

    def _on_zone_selected(self) -> None:
        if self._filling:
            return
        rows = {i.row() for i in self.tw_zones.selectedItems()}
        if not rows:
            return
        item = self.tw_zones.item(min(rows), COL_ID)
        if item is None:
            return
        zid = int(item.text())
        if zid != self._selected_id:
            self._selected_id = zid
            self._fill_rules()
            self._show_problems(self.zones_handler.problems())
            self._update_buttons()
            self._highlight()

    def _on_zone_cell_clicked(self, row: int, col: int) -> None:
        if col in SINGLE_CLICK_EDIT:
            item = self.tw_zones.item(row, col)
            if item is not None:
                self.tw_zones.editItem(item)

    def _on_zone_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        id_item = self.tw_zones.item(item.row(), COL_ID)
        if id_item is None:
            return
        zid = int(id_item.text())
        col = item.column()
        if col == COL_ON:
            self.zones_handler.update_zone(zid, enabled=item.checkState() == _CHECKED)
        elif col == COL_NAME:
            self.zones_handler.update_zone(zid, name=item.text().strip() or f'Zone {zid}')
        elif col in (COL_LIMIT, COL_COMPLIANCE):
            ok, value = _number(item.text())
            valid = ok and value is not None and math.isfinite(value) and (
                value > 0 if col == COL_LIMIT else 0 <= value <= 100)
            if not valid:
                self._on_zones_changed()    # revert the cell
                return
            key = 'limit_kn' if col == COL_LIMIT else 'compliance_pct'
            self.zones_handler.update_zone(zid, **{key: value})

    def _on_draw(self) -> None:
        if self.zones_handler.is_drawing():
            self.zones_handler.cancel_drawing()
            return
        self.zones_handler.start_drawing(self._on_drawn)
        self._update_buttons()
        notifier = getattr(self.omrat, 'notifier', None)
        if notifier is not None:
            try:
                notifier.display_message(self._tr(
                    'Draw the zone on the map: click the corners, right-click to finish.'), duration=8)
            except Exception:  # nosec B110 - the hint is cosmetic
                pass

    def _on_drawn(self, wkt: str | None) -> None:
        try:
            self._update_buttons()
        except RuntimeError:
            return
        if wkt:
            zone = self.zones_handler.add_zone(wkt)
            self.select_zone(zone['id'])

    def _on_remove(self) -> None:
        zone = self.selected_zone()
        if zone is None:
            return
        if not self._testing():
            answer = QMessageBox.question(
                self, self._tr('Remove zone'),
                self._tr('Remove {zone} and its rules?').format(zone=zone_label(zone)))
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.zones_handler.remove_zone(zone['id'])

    # ------------------------------------------------------------------
    # Loading from a file
    # ------------------------------------------------------------------

    def _on_load(self) -> None:
        settings = QSettings()
        start = str(settings.value(LAST_DIR_KEY, '') or '')
        path, _ = QFileDialog.getOpenFileName(
            self, self._tr('Load speed restriction zones'), start,
            self._tr('Vector files (*.shp *.gpkg *.geojson *.json *.kml);;All files (*)'))
        if not path:
            return
        import os
        settings.setValue(LAST_DIR_KEY, os.path.dirname(path))
        layer = QgsVectorLayer(path, 'speed zones', 'ogr')
        if not layer.isValid():
            QMessageBox.warning(self, self._tr('Load speed restriction zones'),
                                self._tr('Could not read {path}.').format(path=path))
            return
        if QgsWkbTypes.geometryType(layer.wkbType()) != QgsWkbTypes.GeometryType.PolygonGeometry:
            QMessageBox.warning(self, self._tr('Load speed restriction zones'),
                                self._tr('The file has no polygons.'))
            return
        choice = self._ask_fields(layer)
        if choice is None:
            return
        self.load_from_layer(layer, *choice)

    def load_from_layer(self, layer: QgsVectorLayer, limit_field: str | None, name_field: str | None,
                        default_limit: float = DEFAULT_LIMIT_KN) -> list[dict[str, Any]]:
        """Headless core of *Load from file...*: one zone per polygon."""
        added = self.zones_handler.add_from_layer(layer, limit_field, name_field, default_limit)
        if added:
            self.select_zone(added[0]['id'])
        return added

    def _ask_fields(self, layer: QgsVectorLayer) -> tuple[str | None, str | None, float] | None:
        """Which attribute holds the limit / the name (guessed; asked unless testing)."""
        names = list(layer.fields().names())
        limit_guess, name_guess = guess_field(names, 'limit'), guess_field(names, 'name')
        if self._testing():
            return limit_guess, name_guess, DEFAULT_LIMIT_KN
        dlg = QDialog(self)
        dlg.setWindowTitle(self._tr('Load speed restriction zones'))
        form = QFormLayout(dlg)
        cb_limit, cb_name = QComboBox(), QComboBox()
        cb_limit.addItem(self._tr('(none: the same limit for every zone)'), None)
        cb_name.addItem(self._tr('(none)'), None)
        for n in names:
            cb_limit.addItem(n, n)
            cb_name.addItem(n, n)
        if limit_guess:
            cb_limit.setCurrentIndex(cb_limit.findData(limit_guess))
        if name_guess:
            cb_name.setCurrentIndex(cb_name.findData(name_guess))
        sp = QDoubleSpinBox()
        sp.setRange(0.1, 60.0)
        sp.setDecimals(1)
        sp.setSuffix(' kn')
        sp.setValue(DEFAULT_LIMIT_KN)
        form.addRow(self._tr('Speed limit from attribute'), cb_limit)
        form.addRow(self._tr('Limit when the attribute is empty'), sp)
        form.addRow(self._tr('Zone name from attribute'), cb_name)
        form.addRow(QLabel(self._tr('{n} feature(s) in the file.').format(n=layer.featureCount())))
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(dlg.accept)
        box.rejected.connect(dlg.reject)
        form.addRow(box)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return cb_limit.currentData(), cb_name.currentData(), float(sp.value())

    # ------------------------------------------------------------------
    # Rules table
    # ------------------------------------------------------------------

    def _categories(self) -> tuple[list[str], list[float], list[float]]:
        cats = self.zones_handler.ship_categories()
        lows, highs = length_boundaries(length_class_bounds(cats.get('length_intervals')))
        return list(cats.get('types') or []), lows, highs

    def _fill_rules(self) -> None:
        zone = self.selected_zone()
        rules = list(zone.get('rules') or []) if zone else []
        types, lows, highs = self._categories()
        self._filling = True
        try:
            self.tw_rules.setRowCount(len(rules))
            for row, rule in enumerate(rules):
                self.tw_rules.setCellWidget(row, RCOL_TYPE, self._type_combo(types, rule.get('ship_type')))
                self.tw_rules.setCellWidget(row, RCOL_FROM, self._length_combo(lows, rule.get('length_min')))
                self.tw_rules.setCellWidget(row, RCOL_TO, self._length_combo(highs, rule.get('length_max')))
                restricted = QTableWidgetItem(self._tr('yes') if rule.get('restricted', True) else self._tr('exempt'))
                restricted.setFlags(_CHECKABLE)
                restricted.setCheckState(_CHECKED if rule.get('restricted', True) else _UNCHECKED)
                self.tw_rules.setItem(row, RCOL_RESTRICTED, restricted)
                self.tw_rules.setItem(row, RCOL_LIMIT, self._item(_num_text(rule.get('limit_kn')), True))
                self.tw_rules.setItem(row, RCOL_COMPLIANCE, self._item(_num_text(rule.get('compliance_pct')), True))
        finally:
            self._filling = False

    def _type_combo(self, types: list[str], value: int | None) -> QComboBox:
        cb = QComboBox()
        cb.addItem(self._tr('All ship types'), None)
        for i, name in enumerate(types):
            cb.addItem(str(name), i)
        if value is not None:
            idx = cb.findData(int(value))
            if idx < 0:
                cb.addItem(self._tr('Ship type {n} (not in the project)').format(n=value), int(value))
                idx = cb.count() - 1
            cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(self._on_rule_widget_changed)
        return cb

    def _length_combo(self, values: list[float], value: float | None) -> QComboBox:
        cb = QComboBox()
        cb.addItem(self._tr('Any'), None)
        for v in values:
            cb.addItem(f'{v:g}', float(v))
        if value is not None:
            idx = next((i for i in range(1, cb.count()) if abs(cb.itemData(i) - float(value)) < 1e-6), -1)
            if idx < 0:
                cb.addItem(self._tr('{v:g} (not a class boundary)').format(v=float(value)), float(value))
                idx = cb.count() - 1
            cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(self._on_rule_widget_changed)
        return cb

    def rules_from_table(self) -> list[dict[str, Any]] | None:
        """The rules as shown; ``None`` when a number cell is not valid."""
        rules = []
        for row in range(self.tw_rules.rowCount()):
            cells = []
            for col, check in ((RCOL_LIMIT, lambda v: v > 0), (RCOL_COMPLIANCE, lambda v: 0 <= v <= 100)):
                item = self.tw_rules.item(row, col)
                text = item.text().strip() if item is not None else ''
                if not text:
                    cells.append(None)
                    continue
                ok, value = _number(text)
                if not ok or value is None or not math.isfinite(value) or not check(value):
                    return None
                cells.append(value)
            restricted = self.tw_rules.item(row, RCOL_RESTRICTED)
            rules.append(new_rule(
                ship_type=self.tw_rules.cellWidget(row, RCOL_TYPE).currentData(),
                length_min=self.tw_rules.cellWidget(row, RCOL_FROM).currentData(),
                length_max=self.tw_rules.cellWidget(row, RCOL_TO).currentData(),
                restricted=restricted is None or restricted.checkState() == _CHECKED,
                limit_kn=cells[0], compliance_pct=cells[1],
            ))
        return rules

    def _store_rules(self, rules: list[dict[str, Any]] | None = None) -> None:
        zone = self.selected_zone()
        if zone is None:
            return
        rules = self.rules_from_table() if rules is None else rules
        if rules is None:
            self._on_zones_changed()     # revert the invalid cell
            return
        self.zones_handler.update_zone(zone['id'], rules=rules)

    def _on_rule_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        if item.column() == RCOL_RESTRICTED:
            self._filling = True
            item.setText(self._tr('yes') if item.checkState() == _CHECKED else self._tr('exempt'))
            self._filling = False
        self._store_rules()

    def _on_rule_widget_changed(self, *_args) -> None:
        if not self._filling:
            self._store_rules()

    def _current_rule_row(self) -> int:
        rows = {i.row() for i in self.tw_rules.selectedItems()}
        if rows:
            return min(rows)
        return self.tw_rules.currentRow()

    def add_rule(self, rule: dict[str, Any] | None = None) -> None:
        zone = self.selected_zone()
        if zone is None:
            return
        rules = list(zone.get('rules') or []) + [rule or new_rule()]
        self._store_rules(rules)

    def remove_rule(self, row: int | None = None) -> None:
        zone = self.selected_zone()
        row = self._current_rule_row() if row is None else row
        if zone is None or not 0 <= row < len(zone.get('rules') or []):
            return
        rules = list(zone['rules'])
        del rules[row]
        self._store_rules(rules)

    def move_rule(self, step: int, row: int | None = None) -> None:
        zone = self.selected_zone()
        row = self._current_rule_row() if row is None else row
        rules = list((zone or {}).get('rules') or [])
        target = row + step
        if zone is None or not 0 <= row < len(rules) or not 0 <= target < len(rules):
            return
        rules[row], rules[target] = rules[target], rules[row]
        self._store_rules(rules)
        QTimer.singleShot(0, lambda: self._select_rule_row(target))

    def _select_rule_row(self, row: int) -> None:
        try:
            if 0 <= row < self.tw_rules.rowCount():
                self.tw_rules.selectRow(row)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------

    def _canvas(self):
        iface = getattr(self.omrat, 'iface', None)
        try:
            return iface.mapCanvas() if iface is not None else None
        except Exception:  # nosec B110 - headless
            return None

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

    def highlighted(self) -> int:
        """Number of highlight bands on the canvas (for tests)."""
        return len(self._bands)

    def _highlight(self) -> None:
        """The selected zone (outline) and the parts of the legs inside it."""
        self._clear_bands()
        canvas = self._canvas()
        zone = self.selected_zone()
        if canvas is None or zone is None:
            return
        geom = QgsGeometry.fromWkt(zone.get('wkt') or '')
        if geom is None or geom.isNull() or geom.isEmpty():
            return
        crs = QgsCoordinateReferenceSystem('EPSG:4326')
        outline = QgsRubberBand(canvas, QgsWkbTypes.GeometryType.PolygonGeometry)
        outline.setColor(QColor(217, 95, 2, 200))
        outline.setFillColor(QColor(217, 95, 2, 40))
        outline.setWidth(3)
        outline.setToGeometry(geom, crs)
        self._bands.append(outline)
        for seg in (getattr(self.omrat, 'segment_data', None) or {}).values():
            if not isinstance(seg, dict):
                continue
            line = QgsGeometry.fromWkt(f"LINESTRING({seg.get('Start_Point')}, {seg.get('End_Point')})")
            if line is None or line.isNull():
                continue
            part = line.intersection(geom)
            if part is None or part.isEmpty() or part.length() <= 0:
                continue
            band = QgsRubberBand(canvas, QgsWkbTypes.GeometryType.LineGeometry)
            band.setColor(QColor(255, 220, 0, 220))
            band.setWidth(6)
            band.setToGeometry(part, crs)
            self._bands.append(band)

    def zoom_to_selected(self) -> None:
        canvas = self._canvas()
        zone = self.selected_zone()
        if canvas is None or zone is None:
            return
        geom = QgsGeometry.fromWkt(zone.get('wkt') or '')
        if geom is None or geom.isEmpty():
            return
        rect = geom.boundingBox()
        rect.scale(1.3)
        canvas_crs = canvas.mapSettings().destinationCrs()
        src = QgsCoordinateReferenceSystem('EPSG:4326')
        if canvas_crs.authid() != src.authid():
            try:
                rect = QgsCoordinateTransform(src, canvas_crs, QgsProject.instance()).transformBoundingBox(rect)
            except Exception:  # nosec B110 - zoom in lon/lat instead
                pass
        canvas.setExtent(rect)
        canvas.refresh()

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------

    def reject(self) -> None:
        self.close()

    def closeEvent(self, event) -> None:
        self._cleanup()
        super().closeEvent(event)

    def _cleanup(self) -> None:
        self.zones_handler.remove_listener(self._on_zones_changed)
        if self.zones_handler.is_drawing():
            self.zones_handler.cancel_drawing()
        self._clear_bands()
        if getattr(self.omrat, '_speed_restriction_dlg', None) is self:
            self.omrat._speed_restriction_dlg = None


def open_speed_restriction_zones(omrat: "OMRAT") -> SpeedRestrictionDialog:
    """Menu entry: open the dialog (modeless) or raise the open one."""
    existing = getattr(omrat, '_speed_restriction_dlg', None)
    if existing is not None:
        try:
            existing.reload()
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return existing
        except RuntimeError:
            omrat._speed_restriction_dlg = None
    dlg = SpeedRestrictionDialog(omrat, getattr(omrat, 'main_widget', None))
    # Keep a Python reference, otherwise the modeless dialog is collected.
    omrat._speed_restriction_dlg = dlg
    dlg.show()
    return dlg
