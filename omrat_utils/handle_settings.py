from typing import Any
from qgis.PyQt.QtWidgets import (
    QDialogButtonBox, QTableWidget, QTableWidgetItem,
    QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QHeaderView,
)
from qgis.PyQt.QtCore import Qt

# Qt 6 (QGIS 4) scopes enums under nested classes (Qt.ItemFlag.*,
# QHeaderView.ResizeMode.*).  Qt 5 (QGIS 3) exposes them directly on the
# owner class.  Accept either.
_ITEM_IS_EDITABLE = getattr(Qt, 'ItemIsEditable', None)
if _ITEM_IS_EDITABLE is None:
    _ITEM_IS_EDITABLE = Qt.ItemFlag.ItemIsEditable
_HV_STRETCH = getattr(QHeaderView, 'Stretch', None)
if _HV_STRETCH is None:
    _HV_STRETCH = QHeaderView.ResizeMode.Stretch
_HV_RESIZE_TO_CONTENTS = getattr(QHeaderView, 'ResizeToContents', None)
if _HV_RESIZE_TO_CONTENTS is None:
    _HV_RESIZE_TO_CONTENTS = QHeaderView.ResizeMode.ResizeToContents

from omrat_utils.repair_time import Repair  # noqa: E402
from ui.drift_settings_widget import DriftSettingsWidget  # noqa: E402
from compute.basic_equations import (  # noqa: E402
    SHIP_TYPE_NAMES,
    default_blackout_by_ship_type,
)
from compute.iwrap_defaults import default_drift_values  # noqa: E402
from omrat_utils.number_input import (  # noqa: E402
    format_weight, normalise_weights, parse_decimal, weights_sum_ok,
)

# Wind-rose line edits in compass order (N, NE, E, SE, S, SW, W, NW) and
# the ``drift['rose']`` key each one maps to.
ROSE_FIELDS: tuple[tuple[str, str], ...] = (
    ('leDriftN', '0'), ('leDriftNE', '45'), ('leDriftE', '90'), ('leDriftSE', '135'),
    ('leDriftS', '180'), ('leDriftSW', '225'), ('leDriftW', '270'), ('leDriftNW', '315'),
)


class DriftSettings:
    def __init__(self, parent):
        self.parent = parent
        self.dsw = DriftSettingsWidget(None)
        self.repair = Repair(self)
        # drift.speed is stored in KNOTS throughout (matches IWRAP import at
        # compute/iwrap_convertion.py:1154 and the cascade at
        # compute/drifting_model.py:2084).
        self.drift_values: dict[str, Any] = default_drift_values()
        self.drift_values['blackout_by_ship_type'] = default_blackout_by_ship_type()
        # The blackout-by-ship-type table is created lazily when the dialog
        # is shown (see _ensure_blackout_table).
        self._blackout_table: QTableWidget | None = None

    # ------------------------------------------------------------------
    # Wind rose
    # ------------------------------------------------------------------
    def _rose_widgets(self) -> list[Any]:
        return [getattr(self.dsw, name) for name, _ in ROSE_FIELDS]

    def _read_rose(self) -> list[float]:
        """Return the eight rose percentages as typed (``,`` accepted).

        Raises ``ValueError`` naming the offending field; an empty field
        counts as 0.
        """
        values: list[float] = []
        for name, _ in ROSE_FIELDS:
            widget = getattr(self.dsw, name)
            try:
                values.append(parse_decimal(widget.text(), default=0.0))
            except ValueError:
                raise ValueError(f"{name[7:]}: '{widget.text()}' is not a number") from None
        if any(v < 0 for v in values):
            raise ValueError("directions cannot be negative")
        return values

    def _set_rose_status(self, text: str, error: bool = False) -> None:
        label = getattr(self.dsw, 'lblRoseSum', None)
        if label is None:
            return
        label.setText(text)
        label.setStyleSheet('color: #b00020;' if error else '')

    def check_rose(self) -> bool:
        """**Check sum** button: verify the eight directions sum to 100 %.

        When they do not, every value is scaled proportionally so the
        total becomes exactly 100 % and the fields are rewritten.  Nothing
        is touched while a field is not a valid number; the label says
        which one.  Returns ``True`` when the rose is valid afterwards.
        """
        try:
            values = self._read_rose()
        except ValueError as exc:
            self._set_rose_status(str(exc), error=True)
            return False
        total = sum(values)
        if weights_sum_ok(values):
            self._set_rose_status(f"Sum: {format_weight(total)} % (OK)")
            return True
        scaled = normalise_weights(values)
        for widget, value in zip(self._rose_widgets(), scaled):
            widget.setText(format_weight(value))
        self._set_rose_status(f"Sum was {format_weight(total)} % -> scaled to 100 %")
        return True

    def _rose_fractions(self) -> dict[str, float]:
        """Rose for ``drift_values``: normalised to 100 % and stored as fractions."""
        values = self._read_rose()
        if not weights_sum_ok(values):
            values = normalise_weights(values)
        return {key: v / 100 for (_, key), v in zip(ROSE_FIELDS, values)}

    def _project_type_names(self) -> list[str] | None:
        """Return the PROJECT's ship-type names (from the Ship Categories
        widget) or ``None`` when unavailable.

        The project taxonomy (often the AIS type list, with "Passenger" at
        index 17) generally differs from OMRAT's internal ``SHIP_TYPE_NAMES``
        (passenger classes at 8-11).  The blackout table must follow the
        project taxonomy because the per-type rates are applied by traffic-
        matrix row index in the drifting cascade.
        """
        try:
            scw = self.parent.ship_cat.scw
            tbl = getattr(scw, 'cvTypes', None)
            if tbl is None:
                return None
            names: list[str] = []
            for i in range(tbl.rowCount()):
                it = tbl.item(i, 0)
                names.append(it.text() if it is not None else '')
            return names if any(names) else None
        except Exception:  # nosec B110 B112
            return None

    def _blackout_row_names(self) -> list[str]:
        """Row labels for the blackout table: project taxonomy if available."""
        names = self._project_type_names()
        if names:
            return names
        return [SHIP_TYPE_NAMES[i] for i in sorted(SHIP_TYPE_NAMES)]

    def _ensure_blackout_table(self) -> QTableWidget:
        """Create the per-ship-type blackout-rate tab the first time it's needed.

        Adds a new tab to the existing ``tabWidget`` containing one row per
        ship-type index in the project's taxonomy (falling back to OMRAT's
        internal list) and a "Reset to IWRAP defaults" button.  Does nothing
        on subsequent calls.
        """
        if self._blackout_table is not None:
            return self._blackout_table
        tab_widget = getattr(self.dsw, 'tabWidget', None)
        if tab_widget is None:
            return None  # type: ignore[return-value]

        container = QWidget()
        layout = QVBoxLayout(container)
        label = QLabel(
            "Blackout rate per ship type (events per ship-year).\n"
            "IWRAP defaults: 1.0 for most types, 0.1 for Passenger / Ro-ro / Ro-pax."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        row_names = self._blackout_row_names()
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Ship type", "Blackout rate (/year)"])
        table.setRowCount(len(row_names))
        for row, name in enumerate(row_names):
            name_item = QTableWidgetItem(f"{row}: {name}")
            name_item.setFlags(name_item.flags() & ~_ITEM_IS_EDITABLE)
            table.setItem(row, 0, name_item)
            # Value cell: editable; we fill the actual value in populate_drift().
            table.setItem(row, 1, QTableWidgetItem(""))
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, _HV_STRETCH)
        header.setSectionResizeMode(1, _HV_RESIZE_TO_CONTENTS)
        layout.addWidget(table)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Reset to IWRAP defaults")
        reset_btn.clicked.connect(self._reset_blackout_defaults)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        tab_widget.addTab(container, "Blackout per ship type")
        self._blackout_table = table
        return table

    def _reset_blackout_defaults(self) -> None:
        """Fill the table with the IWRAP-compatible defaults.

        Uses NAME matching against the project taxonomy when available so
        the 0.1 roro_passenger rate lands on the actual passenger rows.
        """
        if self._blackout_table is None:
            return
        names = self._project_type_names()
        defaults = default_blackout_by_ship_type(names)
        for row in range(self._blackout_table.rowCount()):
            value = defaults.get(row, 1.0)
            item = self._blackout_table.item(row, 1)
            if item is None:
                item = QTableWidgetItem("")
                self._blackout_table.setItem(row, 1, item)
            item.setText(f"{value}")

    def _collect_blackout_from_table(self) -> dict[int, float]:
        """Read the per-ship-type blackout rates from the GUI table."""
        if self._blackout_table is None:
            # Table never created (GUI not shown) -- keep the existing dict.
            return dict(self.drift_values.get('blackout_by_ship_type')
                        or default_blackout_by_ship_type(self._project_type_names()))
        out: dict[int, float] = {}
        for row in range(self._blackout_table.rowCount()):
            item = self._blackout_table.item(row, 1)
            txt = item.text().strip() if item is not None else ""
            try:
                val = parse_decimal(txt, default=1.0)
            except Exception:  # nosec B110 B112
                val = 1.0
            out[row] = max(0.0, val)
        return out

    def commit_changes(self):
        # The rose is normalised here as well as by the Check sum button, so
        # OK never stores (or exports to IWRAP) a rose that does not sum to 1.
        rose = self._rose_fractions()
        # GUI field is in knots; store in knots (matches IWRAP-import and cascade).
        speed = parse_decimal(self.dsw.leDriftSpeed.text())
        drift_p = parse_decimal(self.dsw.leDriftProb.text())
        anchor_raw = parse_decimal(self.dsw.leAnchorProb.text())
        # UI is percentage. Keep backward compatibility if user enters 0-1.
        anchor_p = anchor_raw / 100.0 if anchor_raw > 1.0 else anchor_raw
        anchor_p = max(0.0, min(1.0, anchor_p))
        anchor_d = parse_decimal(self.dsw.leAnchorMaxDepth.text())
        start_mode_text = self.dsw.cbStartDriftingFrom.currentText().strip().lower()
        start_from = 'leg_center' if start_mode_text.startswith('leg') else 'distribution_center'
        squat_mode_text = self.dsw.cbSquatMode.currentText().strip().lower()
        if "drift" in squat_mode_text:
            squat_mode = 'drift_speed'
        elif "don" in squat_mode_text:
            squat_mode = 'none'
        else:
            squat_mode = 'average_speed'
        repair: dict[str, str | float | bool] = {'func': self.dsw.leRepairFunc.toPlainText(),
                                                 'std': parse_decimal(self.dsw.leRepairStd.text()),
                                                 'loc': parse_decimal(self.dsw.leRepairLoc.text()),
                                                 'scale': parse_decimal(self.dsw.leRepairScale.text()),
                                                 'use_lognormal': self.dsw.rbLogNormal.isChecked()}
        blackout_by_ship_type = self._collect_blackout_from_table()
        self.drift_values = {'drift_p': drift_p, 'anchor_p': anchor_p, 'anchor_d': anchor_d, 'speed': speed,
                             'start_from': start_from, 'squat_mode': squat_mode, 'rose': rose,
                             'repair': repair,
                             'blackout_by_ship_type': blackout_by_ship_type}
        self.parent.drift_values = self.drift_values

    def discard_changes(self):
        pass

    def unload(self):
        self.dsw.pbTestRepair.clicked.disconnect()
        check_btn = getattr(self.dsw, 'pbCheckRose', None)
        if check_btn is not None:
            try:
                check_btn.clicked.disconnect()
            except (TypeError, RuntimeError):  # nosec B110 - never connected
                pass
        while self.dsw.canRepairViewLay.count():
            item = self.dsw.canRepairViewLay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()  # Properly delete the widget

    def populate_drift(self):
        """Populates the drift fields with the "drift_values" dict """
        rose = self.drift_values['rose']
        for name, key in ROSE_FIELDS:
            getattr(self.dsw, name).setText(format_weight(float(rose[key]) * 100))
        self._set_rose_status(f"Sum: {format_weight(sum(float(v) for v in rose.values()) * 100)} %")
        # drift.speed is stored in knots; display directly.
        self.dsw.leDriftSpeed.setText(f"{round(float(self.drift_values['speed']), 3)}")
        anchor_val = float(self.drift_values.get('anchor_p', 0.7))
        if anchor_val <= 1.0:
            anchor_display = anchor_val * 100.0
        else:
            anchor_display = anchor_val
        self.dsw.leAnchorProb.setText(f"{round(anchor_display, 3)}")
        self.dsw.leAnchorMaxDepth.setText(f"{self.drift_values['anchor_d']}")
        start_from = str(self.drift_values.get('start_from', 'leg_center')).lower()
        start_idx = 0 if start_from == 'leg_center' else 1
        self.dsw.cbStartDriftingFrom.setCurrentIndex(start_idx)
        squat_mode = str(self.drift_values.get('squat_mode', 'average_speed')).lower()
        if squat_mode == 'drift_speed':
            squat_idx = 1
        elif squat_mode == 'none':
            squat_idx = 2
        else:
            squat_idx = 0
        self.dsw.cbSquatMode.setCurrentIndex(squat_idx)
        self.dsw.leDriftProb.setText(f"{self.drift_values['drift_p']}")
        self.dsw.leRepairFunc.setText(f"{self.drift_values['repair']['func']}")
        self.dsw.leRepairStd.setText(f"{self.drift_values['repair']['std']}")
        self.dsw.leRepairLoc.setText(f"{self.drift_values['repair']['loc']}")
        self.dsw.leRepairScale.setText(f"{self.drift_values['repair']['scale']}")
        # Set BOTH radio buttons explicitly.  Qt's auto-exclusive radio group
        # refuses to uncheck the only checked button (rbLogNormal is
        # default-checked in drift_settings.ui), so setChecked(False) on
        # rbLogNormal alone is silently ignored.  Setting the complement
        # forces the intended state.
        use_ln = bool(self.drift_values['repair']['use_lognormal'])
        self.dsw.rbLogNormal.setChecked(use_ln)
        self.dsw.rbUserDefined.setChecked(not use_ln)
        # Blackout-per-ship-type table.  Merge stored values over the defaults
        # so newly-added ship types (if any) get a sensible fallback instead
        # of blank cells.
        table = self._ensure_blackout_table()
        if table is not None:
            stored_raw = self.drift_values.get('blackout_by_ship_type') or {}
            stored: dict[int, float] = {}
            for k, v in stored_raw.items():
                try:
                    stored[int(k)] = float(v)
                except Exception:  # nosec B110 B112
                    continue
            defaults = default_blackout_by_ship_type(self._project_type_names())
            for row in range(table.rowCount()):
                value = stored.get(row, defaults.get(row, 1.0))
                item = table.item(row, 1)
                if item is None:
                    item = QTableWidgetItem("")
                    table.setItem(row, 1, item)
                item.setText(f"{value}")

    def run(self):
        self.populate_drift()
        self.dsw.show()
        # Get the button box
        self.buttonBox = self.dsw.findChild(QDialogButtonBox, 'buttonBox')
        self.dsw.pbTestRepair.clicked.connect(self.repair.test_evaluate)
        self.dsw.rbLogNormal.toggled.connect(self.repair.test_evaluate)
        self.dsw.rbUserDefined.toggled.connect(self.repair.test_evaluate)
        self.dsw.leRepairStd.textChanged.connect(self.repair.test_evaluate)
        self.dsw.leRepairLoc.textChanged.connect(self.repair.test_evaluate)
        self.dsw.leRepairScale.textChanged.connect(self.repair.test_evaluate)
        # The rose is checked / normalised on demand only.  The previous
        # editingFinished auto-adjust rewrote the other seven fields on
        # every focus change, so a rose could not be typed in field by field.
        check_btn = getattr(self.dsw, 'pbCheckRose', None)
        if check_btn is not None:
            check_btn.clicked.connect(self.check_rose)

        # Connect the accepted signal to your custom slot
        self.buttonBox.accepted.connect(self.commit_changes)

        # Optionally, connect the rejected signal to a different slot
        self.buttonBox.rejected.connect(self.discard_changes)
        self.dsw.exec()
