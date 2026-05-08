"""Owner of the oil-spill consequence inputs and dialogs.

Mirrors the role of ``handle_traffic.Traffic`` for the new ``Consequence``
top-level menu: holds the live state (oil_onboard, spill_probability,
spill_fraction, catastrophe_levels), exposes ``run_*`` methods that open
the editing dialogs, and reshapes the matrices when ship categories change.

The handler is constructed from ``omrat.OMRAT.__init__`` and queried by
``gather_data`` on save and ``compute/consequence`` at run time.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from qgis.PyQt import QtCore, QtWidgets
from qgis.PyQt.QtWidgets import (
    QDoubleSpinBox, QHeaderView, QMessageBox, QTableWidget,
    QTableWidgetItem,
)

from omrat_utils.consequence_defaults import (
    ACCIDENT_TYPES,
    SPILL_LEVELS,
    default_catastrophe_levels,
    default_oil_onboard,
    default_spill_fraction,
    default_spill_probability,
    reshape_oil_onboard,
)

if TYPE_CHECKING:
    from omrat import OMRAT


def _qt_align_center() -> int:
    Qt = QtCore.Qt
    try:
        return int(Qt.AlignmentFlag.AlignCenter)
    except AttributeError:
        return int(Qt.AlignCenter)


def _qt_attr(klass, *names: str):
    """Cross PyQt5 / PyQt6: try flat names, then scoped enum paths.

    PyQt5 exposes ``QDialogButtonBox.Reset`` flat; PyQt6 only has it as
    ``QDialogButtonBox.StandardButton.Reset``.  Same idea for
    ``QDialog.Accepted`` -> ``QDialog.DialogCode.Accepted``.
    """
    for name in names:
        if '.' in name:
            obj = klass
            try:
                for part in name.split('.'):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        elif hasattr(klass, name):
            return getattr(klass, name)
    raise AttributeError(
        f"None of {names!r} are attributes of {klass.__name__}"
    )


_BB_RESET = None
_BB_OK = None
_DIALOG_ACCEPTED = None


def _bb_reset():
    global _BB_RESET
    if _BB_RESET is None:
        _BB_RESET = _qt_attr(
            QtWidgets.QDialogButtonBox, 'Reset', 'StandardButton.Reset',
        )
    return _BB_RESET


def _bb_ok():
    global _BB_OK
    if _BB_OK is None:
        _BB_OK = _qt_attr(
            QtWidgets.QDialogButtonBox, 'Ok', 'StandardButton.Ok',
        )
    return _BB_OK


def _dialog_accepted():
    global _DIALOG_ACCEPTED
    if _DIALOG_ACCEPTED is None:
        _DIALOG_ACCEPTED = _qt_attr(
            QtWidgets.QDialog, 'Accepted', 'DialogCode.Accepted',
        )
    return _DIALOG_ACCEPTED


class Consequence:
    """Project-level consequence-input owner."""

    def __init__(self, omrat: "OMRAT") -> None:
        self.omrat = omrat
        # Live state.  Each is replaced wholesale on .omrat load and on
        # successful dialog OK; rejected dialogs leave the existing state
        # untouched.
        self.oil_onboard: list[list[float]] = []
        self.spill_probability: list[list[float]] = default_spill_probability()
        self.spill_fraction: list[list[float]] = default_spill_fraction()
        self.catastrophe_levels: list[dict[str, Any]] = default_catastrophe_levels()
        # Cache the ship-type / length-interval lists used last time we
        # reshaped oil_onboard, so we can detect dimension changes when
        # the dialog is reopened.
        self._last_ship_types: list[str] = []
        self._last_length_intervals: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def _current_ship_categories(self) -> tuple[list[str], list[dict[str, Any]]]:
        """Pull the live ship-type and length-interval lists from the
        Ship Categories widget.  Falls back to whatever we last loaded
        from a .omrat if the widget isn't reachable.
        """
        scw = getattr(getattr(self.omrat, 'ship_cat', None), 'scw', None)
        types: list[str] = []
        intervals: list[dict[str, Any]] = []
        if scw is None:
            return list(self._last_ship_types), list(self._last_length_intervals)
        try:
            for i in range(scw.cvTypes.rowCount()):
                it = scw.cvTypes.item(i, 0)
                txt = it.text() if it is not None else ''
                if txt:
                    types.append(txt)
            for i in range(scw.twLengths.rowCount()):
                it_min = scw.twLengths.item(i, 0)
                it_max = scw.twLengths.item(i, 1)
                smin = it_min.text() if it_min is not None else ''
                smax = it_max.text() if it_max is not None else ''
                if smin == '' and smax == '':
                    continue
                try:
                    vmin: float | str = float(smin)
                except (TypeError, ValueError):
                    vmin = smin
                try:
                    vmax: float | str = float(smax)
                except (TypeError, ValueError):
                    vmax = smax
                intervals.append({
                    'min': vmin,
                    'max': vmax,
                    'label': f'{smin} - {smax}',
                })
        except Exception:
            return list(self._last_ship_types), list(self._last_length_intervals)
        return types, intervals

    def _refresh_oil_onboard_shape(self) -> tuple[list[str], list[dict[str, Any]]]:
        """Reshape ``self.oil_onboard`` to the current ship-category dims.

        Returns the ship-types / length-intervals lists that were used,
        so callers can populate the table headers without duplicating the
        widget read.
        """
        types, intervals = self._current_ship_categories()
        self.oil_onboard = reshape_oil_onboard(self.oil_onboard, types, intervals)
        self._last_ship_types = list(types)
        self._last_length_intervals = list(intervals)
        return types, intervals

    def load_from_dict(
        self,
        block: dict[str, Any],
        ship_categories: dict[str, Any],
    ) -> None:
        """Replace state from a loaded ``.omrat``'s ``consequence`` block.

        Reshapes ``oil_onboard`` to the loaded ship_categories so older
        projects whose ship-type / length-interval lists differ from the
        current defaults still display sensibly.
        """
        types = list(ship_categories.get('types', []))
        intervals = list(ship_categories.get('length_intervals', []))

        self.oil_onboard = reshape_oil_onboard(
            block.get('oil_onboard'), types, intervals,
        )

        spill_prob = block.get('spill_probability')
        if isinstance(spill_prob, list) and len(spill_prob) == len(ACCIDENT_TYPES):
            self.spill_probability = [
                [float(v) for v in row] for row in spill_prob
            ]
        else:
            self.spill_probability = default_spill_probability()

        spill_frac = block.get('spill_fraction')
        if isinstance(spill_frac, list) and len(spill_frac) == len(ACCIDENT_TYPES):
            self.spill_fraction = [
                [float(v) for v in row] for row in spill_frac
            ]
        else:
            self.spill_fraction = default_spill_fraction()

        levels = block.get('catastrophe_levels')
        if isinstance(levels, list) and len(levels) >= 2:
            cleaned: list[dict[str, Any]] = []
            for entry in levels:
                if not isinstance(entry, dict):
                    continue
                try:
                    cleaned.append({
                        'name': str(entry.get('name', '')),
                        'quantity': float(entry.get('quantity', 0.0)),
                    })
                except (TypeError, ValueError):
                    continue
            if len(cleaned) >= 2:
                self.catastrophe_levels = cleaned
            else:
                self.catastrophe_levels = default_catastrophe_levels()
        else:
            self.catastrophe_levels = default_catastrophe_levels()

        self._last_ship_types = list(types)
        self._last_length_intervals = list(intervals)

    # ------------------------------------------------------------------
    # Oil onboard dialog
    # ------------------------------------------------------------------
    def run_oil_onboard_dialog(self) -> None:
        from ui.oil_onboard_widget import OilOnboardWidget

        types, intervals = self._refresh_oil_onboard_shape()
        if not types or not intervals:
            QMessageBox.information(
                self.omrat.main_widget,
                self.omrat.tr('No ship categories'),
                self.omrat.tr(
                    'Define ship types and length intervals under '
                    'Settings -> Ship Categories before editing oil onboard.'
                ),
            )
            return

        dlg = OilOnboardWidget(self.omrat.main_widget)
        self._populate_oil_onboard_table(
            dlg.twOilOnboard, types, intervals, self.oil_onboard,
        )
        # Reset button restores defaults *for the current dimensions*.
        reset_btn = dlg.buttonBox.button(_bb_reset())
        if reset_btn is not None:
            reset_btn.clicked.connect(lambda: self._populate_oil_onboard_table(
                dlg.twOilOnboard, types, intervals,
                default_oil_onboard(types, intervals),
            ))

        if dlg.exec() != _dialog_accepted():
            return
        self.oil_onboard = self._read_matrix(dlg.twOilOnboard)

    @staticmethod
    def _populate_oil_onboard_table(
        tw: QTableWidget,
        types: list[str],
        intervals: list[dict[str, Any]],
        values: list[list[float]],
    ) -> None:
        col_labels = [str(itv.get('label', '')) for itv in intervals]
        tw.clear()
        tw.setRowCount(len(types))
        tw.setColumnCount(len(col_labels))
        tw.setHorizontalHeaderLabels(col_labels)
        tw.setVerticalHeaderLabels(types)
        tw.verticalHeader().setVisible(True)
        try:
            tw.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch,
            )
        except AttributeError:
            tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for r in range(len(types)):
            for c in range(len(col_labels)):
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 10_000_000.0)
                spin.setDecimals(2)
                spin.setSingleStep(10.0)
                try:
                    spin.setValue(float(values[r][c]))
                except (IndexError, TypeError, ValueError):
                    spin.setValue(0.0)
                tw.setCellWidget(r, c, spin)

    # ------------------------------------------------------------------
    # Spill probability dialog (sum-to-100 enforced)
    # ------------------------------------------------------------------
    def run_spill_probability_dialog(self) -> None:
        from ui.spill_probability_widget import SpillProbabilityWidget

        dlg = SpillProbabilityWidget(self.omrat.main_widget)
        self._populate_spill_table(
            dlg.twSpillProbability, self.spill_probability,
            cell_max=100.0, decimals=2, single_step=1.0,
        )

        def _refresh_status() -> None:
            sums = self._row_sums(dlg.twSpillProbability)
            bad = [i for i, s in enumerate(sums) if abs(s - 100.0) > 0.05]
            if bad:
                msg = 'Rows not summing to 100: ' + ', '.join(
                    f'{ACCIDENT_TYPES[i]} ({sums[i]:.2f})' for i in bad
                )
                dlg.lblRowSumStatus.setText(msg)
                dlg.lblRowSumStatus.setStyleSheet('color: #b71c1c;')
            else:
                dlg.lblRowSumStatus.setText('All rows sum to 100%.')
                dlg.lblRowSumStatus.setStyleSheet('color: #1b5e20;')

        for r in range(dlg.twSpillProbability.rowCount()):
            for c in range(dlg.twSpillProbability.columnCount()):
                w = dlg.twSpillProbability.cellWidget(r, c)
                if isinstance(w, QDoubleSpinBox):
                    w.valueChanged.connect(lambda *_: _refresh_status())
        _refresh_status()

        reset_btn = dlg.buttonBox.button(_bb_reset())
        if reset_btn is not None:
            def _do_reset() -> None:
                self._populate_spill_table(
                    dlg.twSpillProbability, default_spill_probability(),
                    cell_max=100.0, decimals=2, single_step=1.0,
                )
                _refresh_status()
            reset_btn.clicked.connect(_do_reset)

        # Block OK if any row diverges from 100% by more than 0.05.
        ok_btn = dlg.buttonBox.button(_bb_ok())
        if ok_btn is not None:
            def _validate_and_accept() -> None:
                sums = self._row_sums(dlg.twSpillProbability)
                if any(abs(s - 100.0) > 0.05 for s in sums):
                    QMessageBox.warning(
                        dlg, self.omrat.tr('Row sums must equal 100%'),
                        self.omrat.tr(
                            'Each accident-category row must sum to 100%. '
                            'Adjust the highlighted rows before saving.'
                        ),
                    )
                    return
                dlg.accept()
            try:
                ok_btn.clicked.disconnect()
            except TypeError:
                pass
            ok_btn.clicked.connect(_validate_and_accept)

        if dlg.exec() != _dialog_accepted():
            return
        self.spill_probability = self._read_matrix(dlg.twSpillProbability)

    # ------------------------------------------------------------------
    # Spill fraction dialog
    # ------------------------------------------------------------------
    def run_spill_fraction_dialog(self) -> None:
        from ui.spill_fraction_widget import SpillFractionWidget

        dlg = SpillFractionWidget(self.omrat.main_widget)
        self._populate_spill_table(
            dlg.twSpillFraction, self.spill_fraction,
            cell_max=100.0, decimals=2, single_step=1.0,
        )
        reset_btn = dlg.buttonBox.button(_bb_reset())
        if reset_btn is not None:
            reset_btn.clicked.connect(lambda: self._populate_spill_table(
                dlg.twSpillFraction, default_spill_fraction(),
                cell_max=100.0, decimals=2, single_step=1.0,
            ))

        if dlg.exec() != _dialog_accepted():
            return
        self.spill_fraction = self._read_matrix(dlg.twSpillFraction)

    @staticmethod
    def _populate_spill_table(
        tw: QTableWidget,
        values: list[list[float]],
        *,
        cell_max: float,
        decimals: int,
        single_step: float,
    ) -> None:
        tw.clear()
        tw.setRowCount(len(ACCIDENT_TYPES))
        tw.setColumnCount(len(SPILL_LEVELS))
        tw.setHorizontalHeaderLabels(list(SPILL_LEVELS))
        tw.setVerticalHeaderLabels(list(ACCIDENT_TYPES))
        tw.verticalHeader().setVisible(True)
        try:
            tw.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch,
            )
        except AttributeError:
            tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for r in range(len(ACCIDENT_TYPES)):
            for c in range(len(SPILL_LEVELS)):
                spin = QDoubleSpinBox()
                spin.setRange(0.0, cell_max)
                spin.setDecimals(decimals)
                spin.setSingleStep(single_step)
                try:
                    spin.setValue(float(values[r][c]))
                except (IndexError, TypeError, ValueError):
                    spin.setValue(0.0)
                tw.setCellWidget(r, c, spin)

    @staticmethod
    def _row_sums(tw: QTableWidget) -> list[float]:
        sums: list[float] = []
        for r in range(tw.rowCount()):
            total = 0.0
            for c in range(tw.columnCount()):
                w = tw.cellWidget(r, c)
                if isinstance(w, QDoubleSpinBox):
                    total += float(w.value())
            sums.append(total)
        return sums

    @staticmethod
    def _read_matrix(tw: QTableWidget) -> list[list[float]]:
        out: list[list[float]] = []
        for r in range(tw.rowCount()):
            row: list[float] = []
            for c in range(tw.columnCount()):
                w = tw.cellWidget(r, c)
                if isinstance(w, QDoubleSpinBox):
                    row.append(float(w.value()))
                else:
                    row.append(0.0)
            out.append(row)
        return out

    # ------------------------------------------------------------------
    # Catastrophe levels dialog
    # ------------------------------------------------------------------
    def run_catastrophe_levels_dialog(self) -> None:
        from ui.catastrophe_levels_widget import CatastropheLevelsWidget

        dlg = CatastropheLevelsWidget(self.omrat.main_widget)
        self._populate_catastrophe_table(
            dlg.twCatastropheLevels, self.catastrophe_levels,
        )

        def _add_row() -> None:
            tw = dlg.twCatastropheLevels
            row = tw.rowCount()
            tw.insertRow(row)
            tw.setItem(row, 0, QTableWidgetItem('New level'))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 10_000_000.0)
            spin.setDecimals(2)
            spin.setSingleStep(50.0)
            tw.setCellWidget(row, 1, spin)

        def _remove_row() -> None:
            tw = dlg.twCatastropheLevels
            if tw.rowCount() <= 2:
                QMessageBox.information(
                    dlg, self.omrat.tr('Minimum two rows'),
                    self.omrat.tr(
                        'Catastrophe definitions require at least two levels.'
                    ),
                )
                return
            sel = sorted({idx.row() for idx in tw.selectedIndexes()}, reverse=True)
            if not sel:
                tw.removeRow(tw.rowCount() - 1)
                return
            for r in sel:
                if tw.rowCount() <= 2:
                    break
                tw.removeRow(r)

        dlg.pbAddLevel.clicked.connect(_add_row)
        dlg.pbRemoveLevel.clicked.connect(_remove_row)

        reset_btn = dlg.buttonBox.button(_bb_reset())
        if reset_btn is not None:
            reset_btn.clicked.connect(lambda: self._populate_catastrophe_table(
                dlg.twCatastropheLevels, default_catastrophe_levels(),
            ))

        ok_btn = dlg.buttonBox.button(_bb_ok())
        if ok_btn is not None:
            def _validate_and_accept() -> None:
                if dlg.twCatastropheLevels.rowCount() < 2:
                    QMessageBox.warning(
                        dlg, self.omrat.tr('Minimum two rows'),
                        self.omrat.tr(
                            'Catastrophe definitions require at least two levels.'
                        ),
                    )
                    return
                dlg.accept()
            try:
                ok_btn.clicked.disconnect()
            except TypeError:
                pass
            ok_btn.clicked.connect(_validate_and_accept)

        if dlg.exec() != _dialog_accepted():
            return
        self.catastrophe_levels = self._read_catastrophe_table(dlg.twCatastropheLevels)

    @staticmethod
    def _populate_catastrophe_table(
        tw: QTableWidget,
        levels: list[dict[str, Any]],
    ) -> None:
        tw.clear()
        tw.setColumnCount(2)
        tw.setHorizontalHeaderLabels(['Name', 'Quantity (m^3)'])
        tw.setRowCount(len(levels))
        try:
            tw.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch,
            )
        except AttributeError:
            tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tw.verticalHeader().setVisible(False)
        for r, level in enumerate(levels):
            tw.setItem(r, 0, QTableWidgetItem(str(level.get('name', ''))))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 10_000_000.0)
            spin.setDecimals(2)
            spin.setSingleStep(50.0)
            try:
                spin.setValue(float(level.get('quantity', 0.0)))
            except (TypeError, ValueError):
                spin.setValue(0.0)
            tw.setCellWidget(r, 1, spin)

    @staticmethod
    def _read_catastrophe_table(tw: QTableWidget) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in range(tw.rowCount()):
            name_item = tw.item(r, 0)
            name = name_item.text() if name_item is not None else f'Level {r + 1}'
            spin = tw.cellWidget(r, 1)
            qty = float(spin.value()) if isinstance(spin, QDoubleSpinBox) else 0.0
            out.append({'name': name, 'quantity': qty})
        return out
