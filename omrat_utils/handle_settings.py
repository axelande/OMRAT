from typing import Any
from qgis.PyQt.QtWidgets import (
    QDialogButtonBox, QLineEdit, QTableWidget, QTableWidgetItem,
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

from omrat_utils.repair_time import Repair
from ui.drift_settings_widget import DriftSettingsWidget
from compute.basic_equations import (
    SHIP_TYPE_NAMES,
    default_blackout_by_ship_type,
)

class DriftSettings:
    def __init__(self, parent):
        self.parent = parent
        self.dsw = DriftSettingsWidget(None)
        self.repair = Repair(self)
        rose = {'0': .125, '45': .125, '90': .125, '135': .125, '180': .125, '225': .125, '270': .125, '315': .125}
        repair: dict[str,str|float|bool] = {'func': "",
                  'std': .95,
                  'loc': .2,
                  'scale': .85,
                  'use_lognormal': True}
        # Per-ship-type blackout rate (events/ship-year).  IWRAP-compatible
        # defaults: 1.0 for most types, 0.1 for RoRo / Passenger.
        blackout_by_ship_type = default_blackout_by_ship_type()
        # This is set here as default values, however it is overwritten while loading user data.
        # drift.speed is stored in KNOTS throughout (matches IWRAP import at
        # compute/iwrap_convertion.py:1154 and the cascade at
        # compute/drifting_model.py:2084).
        self.drift_values:dict[str, Any] = {'drift_p': 1, 'anchor_p': .70,'anchor_d': 7, 'speed': 1.0,
                            'start_from': 'leg_center',
                            'squat_mode': 'average_speed',
                                            'rose': rose, 'repair': repair,
                                            'blackout_by_ship_type': blackout_by_ship_type}
        # The blackout-by-ship-type table is created lazily when the dialog
        # is shown (see _ensure_blackout_table).
        self._blackout_table: QTableWidget | None = None
        
    def adjust_directions(self, changed_widget: QLineEdit) -> None:
        widgets: list[Any] = [self.dsw.leDriftN, self.dsw.leDriftNE, self.dsw.leDriftNW, self.dsw.leDriftE, 
                              self.dsw.leDriftSE, self.dsw.leDriftS, self.dsw.leDriftSW, self.dsw.leDriftW]
        
        total_weight = 100
        changed_value = float(changed_widget.text())

        # Calculate the remaining weight
        remaining_weight = total_weight - changed_value

        # Distribute the remaining weight proportionally among the other widgets
        other_widgets = [w for w in widgets if w != changed_widget]
        other_values = [
            float(w.text()) if hasattr(w, 'text') and w.text() != '' else w.value() if hasattr(w, 'value') else 0
            for w in other_widgets
        ]
        other_total = sum(other_values)

        if other_total == 0:
            # If all other values are zero, distribute equally
            for w in other_widgets:
                w.setText(str(remaining_weight / len(other_widgets)))
        else:
            # Adjust the other values proportionally
            for w, value in zip(other_widgets, other_values):
                adjusted_value = (value / other_total) * remaining_weight
                w.setText(str(round(adjusted_value, 2)))
        # Ensure the total sum is exactly 100
        self.ensure_total_sum(widgets)

    def ensure_total_sum(self, widgets:list[QLineEdit]):
        """Ensure the total sum of weights equals 100."""
        total = sum(
            float(w.text()) if w.text() != '' else 0
            for w in widgets
        )
        difference = 100 - total

        # Adjust the last widget to make the total exactly 100
        last_widget = widgets[-1]
        last_value = float(last_widget.text()) if last_widget.text() != '' else 0
        last_widget.setText(str(last_value + difference))
        
    def _ensure_blackout_table(self) -> QTableWidget:
        """Create the per-ship-type blackout-rate tab the first time it's needed.

        Adds a new tab to the existing ``tabWidget`` containing a 21-row
        QTableWidget (one row per OMRAT ship-type index) and a
        "Reset to IWRAP defaults" button.  Does nothing on subsequent calls.
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

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Ship type", "Blackout rate (/year)"])
        table.setRowCount(len(SHIP_TYPE_NAMES))
        for row, idx in enumerate(sorted(SHIP_TYPE_NAMES)):
            name_item = QTableWidgetItem(f"{idx}: {SHIP_TYPE_NAMES[idx]}")
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
        """Fill the table with the IWRAP-compatible defaults."""
        if self._blackout_table is None:
            return
        defaults = default_blackout_by_ship_type()
        for row, idx in enumerate(sorted(SHIP_TYPE_NAMES)):
            value = defaults.get(idx, 1.0)
            item = self._blackout_table.item(row, 1)
            if item is None:
                item = QTableWidgetItem("")
                self._blackout_table.setItem(row, 1, item)
            item.setText(f"{value}")

    def _collect_blackout_from_table(self) -> dict[int, float]:
        """Read the per-ship-type blackout rates from the GUI table."""
        if self._blackout_table is None:
            # Table never created (GUI not shown) -- keep the existing dict.
            return dict(self.drift_values.get('blackout_by_ship_type') or default_blackout_by_ship_type())
        out: dict[int, float] = {}
        for row, idx in enumerate(sorted(SHIP_TYPE_NAMES)):
            item = self._blackout_table.item(row, 1)
            txt = item.text().strip() if item is not None else ""
            try:
                val = float(txt) if txt else 1.0
            except Exception:
                val = 1.0
            out[int(idx)] = max(0.0, val)
        return out

    def commit_changes(self):
        n = float(self.dsw.leDriftN.text()) / 100
        ne = float(self.dsw.leDriftNE.text()) / 100
        e = float(self.dsw.leDriftE.text()) / 100
        se = float(self.dsw.leDriftSE.text()) / 100
        s = float(self.dsw.leDriftS.text()) / 100
        sw = float(self.dsw.leDriftSW.text()) / 100
        w = float(self.dsw.leDriftW.text()) / 100
        nw = float(self.dsw.leDriftNW.text()) / 100
        rose = {'0': n, '45': ne, '90': e, '135': se, '180': s, '225': sw, '270': w, '315': nw}
        # GUI field is in knots; store in knots (matches IWRAP-import and cascade).
        speed = float(self.dsw.leDriftSpeed.text())
        drift_p = float(self.dsw.leDriftProb.text())
        anchor_raw = float(self.dsw.leAnchorProb.text())
        # UI is percentage. Keep backward compatibility if user enters 0-1.
        anchor_p = anchor_raw / 100.0 if anchor_raw > 1.0 else anchor_raw
        anchor_p = max(0.0, min(1.0, anchor_p))
        anchor_d = float(self.dsw.leAnchorMaxDepth.text())
        start_mode_text = self.dsw.cbStartDriftingFrom.currentText().strip().lower()
        start_from = 'leg_center' if start_mode_text.startswith('leg') else 'distribution_center'
        squat_mode_text = self.dsw.cbSquatMode.currentText().strip().lower()
        if "drift" in squat_mode_text:
            squat_mode = 'drift_speed'
        elif "don" in squat_mode_text:
            squat_mode = 'none'
        else:
            squat_mode = 'average_speed'
        repair: dict[str,str|float|bool] = {'func': self.dsw.leRepairFunc.toPlainText(),
                  'std': float(self.dsw.leRepairStd.text()),
                  'loc': float(self.dsw.leRepairLoc.text()),
                  'scale': float(self.dsw.leRepairScale.text()),
                  'use_lognormal': self.dsw.rbLogNormal.isChecked()}
        blackout_by_ship_type = self._collect_blackout_from_table()
        self.drift_values = {'drift_p': drift_p, 'anchor_p':anchor_p,'anchor_d':anchor_d, 'speed':speed,
                     'start_from': start_from, 'squat_mode': squat_mode, 'rose': rose,
                             'repair': repair,
                             'blackout_by_ship_type': blackout_by_ship_type}
        self.parent.drift_values = self.drift_values
        
    def discard_changes(self):
        pass
    
    def unload(self):
        self.dsw.pbTestRepair.clicked.disconnect()
        widgets: list[Any] = [self.dsw.leDriftN, self.dsw.leDriftNE, self.dsw.leDriftNW, self.dsw.leDriftE, 
                              self.dsw.leDriftSE, self.dsw.leDriftS, self.dsw.leDriftSW, self.dsw.leDriftW]
        for widget in widgets:
            if hasattr(widget, 'editingFinished'):
                widget.editingFinished.disconnect()
            elif hasattr(widget, 'leaveEvent'):
                widget.leaveEvent.disconnect()
        while self.dsw.canRepairViewLay.count():
            item = self.dsw.canRepairViewLay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()  # Properly delete the widget

    def populate_drift(self):
        """Populates the drift fields with the "drift_values" dict """
        self.dsw.leDriftN.setText(f"{self.drift_values['rose']['0'] * 100}")
        self.dsw.leDriftNE.setText(f"{self.drift_values['rose']['45'] * 100}")
        self.dsw.leDriftE.setText(f"{self.drift_values['rose']['90'] * 100}")
        self.dsw.leDriftSE.setText(f"{self.drift_values['rose']['135'] * 100}")
        self.dsw.leDriftS.setText(f"{self.drift_values['rose']['180'] * 100}")
        self.dsw.leDriftSW.setText(f"{self.drift_values['rose']['225'] * 100}")
        self.dsw.leDriftW.setText(f"{self.drift_values['rose']['270'] * 100}")
        self.dsw.leDriftNW.setText(f"{self.drift_values['rose']['315'] * 100}")
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
                except Exception:
                    continue
            defaults = default_blackout_by_ship_type()
            for row, idx in enumerate(sorted(SHIP_TYPE_NAMES)):
                value = stored.get(idx, defaults.get(idx, 1.0))
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
        widgets: list[Any] = [self.dsw.leDriftN, self.dsw.leDriftNE, self.dsw.leDriftNW, self.dsw.leDriftE, 
                        self.dsw.leDriftSE, self.dsw.leDriftS, self.dsw.leDriftSW, self.dsw.leDriftW]
        for widget in widgets:
            if hasattr(widget, 'editingFinished'):
                widget.editingFinished.connect(lambda w=widget: self.adjust_directions(w))
            elif hasattr(widget, 'leaveEvent'):
                widget.leaveEvent.connect(lambda w=widget: self.adjust_directions(w))
                
        # Connect the accepted signal to your custom slot
        self.buttonBox.accepted.connect(self.commit_changes)
        
        # Optionally, connect the rejected signal to a different slot
        self.buttonBox.rejected.connect(self.discard_changes)
        self.dsw.exec()
