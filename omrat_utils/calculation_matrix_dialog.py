"""Run Analysis tab -> **Calculation matrix...**: pick the model parts
**Run model** calculates.

The selection lives on ``OMRAT.run_phases`` for the session (a fresh
QGIS session runs everything again) and is handed to
``CalculationTask``.  The phase vocabulary and the "skipped means empty,
not zero" rule are in :mod:`compute.run_phases`.
"""
from __future__ import annotations

from typing import Iterable

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from compute.run_phases import (
    PHASE_LABELS, PHASES, describe_selection, is_partial, normalize_phases,
)

# What each row produces (the result-table rows of ``PHASE_TOTALS``).
_ACCIDENT_TYPES = {
    'drifting': 'Drifting allision and drifting grounding',
    'collision': 'Overtaking, head-on, crossing, merging and bend collisions',
    'powered_grounding': 'Powered grounding (Cat I + Cat II)',
    'powered_allision': 'Powered allision (Cat I + Cat II)',
}
_NOTE = (
    "Parts that are not ticked are left out of the run: their rows in the "
    "result table stay empty, and the run's report, GeoPackage and history "
    "hold only the ticked parts.  Oil-spill consequences are computed from "
    "the ticked parts."
)


class CalculationMatrixDialog(QDialog):
    """One row per model part with a Calculate check box."""

    def __init__(self, phases: Iterable[str] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('Calculation matrix'))
        selected = set(normalize_phases(phases))

        self.table = QTableWidget(len(PHASES), 2, self)
        self.table.setHorizontalHeaderLabels([self.tr('Calculate'), self.tr('Accident types')])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for row, phase in enumerate(PHASES):
            item = QTableWidgetItem(self.tr(PHASE_LABELS[phase]))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if phase in selected else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, phase)
            self.table.setItem(row, 0, item)
            types_item = QTableWidgetItem(self.tr(_ACCIDENT_TYPES[phase]))
            types_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 1, types_item)
        self.table.resizeRowsToContents()
        self.table.resizeColumnToContents(1)
        self.table.setCurrentCell(-1, -1)
        self.table.setMinimumWidth(
            self.table.columnWidth(0) + self.table.columnWidth(1) + 24)
        self.table.setFixedHeight(
            self.table.horizontalHeader().height()
            + sum(self.table.rowHeight(r) for r in range(self.table.rowCount())) + 4)

        note = QLabel(self.tr(_NOTE), self)
        note.setWordWrap(True)

        self.pb_all = QPushButton(self.tr('Select all'), self)
        self.pb_none = QPushButton(self.tr('Select none'), self)
        self.pb_all.clicked.connect(lambda: self._set_all(True))
        self.pb_none.clicked.connect(lambda: self._set_all(False))
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(self.pb_all)
        row.addWidget(self.pb_none)
        row.addStretch(1)
        row.addWidget(self.buttons)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(self.tr('Tick the model parts Run model calculates.'), self))
        layout.addWidget(self.table)
        layout.addWidget(note)
        layout.addLayout(row)

        self.table.itemChanged.connect(lambda *_: self._update_ok())
        self._update_ok()

    def selected_phases(self) -> tuple[str, ...]:
        picked = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                picked.append(item.data(Qt.ItemDataRole.UserRole))
        return normalize_phases(picked)

    def _set_all(self, on: bool) -> None:
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(state)

    def _update_ok(self) -> None:
        # Nothing ticked = nothing to run.
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(bool(self.selected_phases()))


class CalculationMatrixMixin:
    """``pbCalculationMatrix`` wiring + the session's part selection."""

    def _setup_calculation_matrix_button(self) -> None:
        btn = getattr(self.main_widget, 'pbCalculationMatrix', None)
        if btn is not None:
            btn.clicked.connect(self.open_calculation_matrix)
        self._refresh_run_phases_ui()

    def open_calculation_matrix(self) -> None:
        dlg = CalculationMatrixDialog(getattr(self, 'run_phases', None), self.main_widget)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.set_run_phases(dlg.selected_phases())

    def set_run_phases(self, phases: Iterable[str] | None) -> None:
        self.run_phases = normalize_phases(phases)
        self._refresh_run_phases_ui()

    def _run_model_tooltip(self) -> str:
        """Run model's tooltip while it is enabled ('' for a full run)."""
        phases = getattr(self, 'run_phases', None)
        if not is_partial(phases):
            return ''
        return self.tr('Calculates only: {parts}.  Change it with Calculation matrix...').format(
            parts=describe_selection(phases))

    def _refresh_run_phases_ui(self) -> None:
        """A partial selection shows on Run model itself, so a run that
        skips parts is never started by accident."""
        phases = normalize_phases(getattr(self, 'run_phases', None))
        run_btn = getattr(self.main_widget, 'pbRunModel', None)
        if run_btn is not None:
            if is_partial(phases):
                run_btn.setText(self.tr('Run model ({n} of {total} parts)').format(
                    n=len(phases), total=len(PHASES)))
            else:
                run_btn.setText(self.tr('Run model'))
        matrix_btn = getattr(self.main_widget, 'pbCalculationMatrix', None)
        if matrix_btn is not None:
            matrix_btn.setToolTip(self.tr(
                'Choose which model parts Run model calculates (now: {parts})').format(
                    parts=describe_selection(phases)))
        update = getattr(self, '_update_run_model_enabled', None)
        if callable(update):
            update()
