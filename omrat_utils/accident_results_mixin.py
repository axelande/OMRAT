"""Accident-results table + per-row View dispatcher, factored out of
``omrat.OMRAT``.

Owns the ``TWAccidentResults`` table on the Run Analysis tab plus the
View-button slots that open the interactive visualiser for the
currently-selected run in ``TWPreviousRuns``:

* :meth:`AccidentResultsMixin._setup_accident_results_table` builds
  the table, creates the legacy ``LEP*`` line-edits as hidden widgets
  (so existing compute / test code that calls ``setText`` still works),
  and wires the per-row ``View`` button.
* :meth:`AccidentResultsMixin._dispatch_view` is the single entry-point
  every ``show_*`` slot delegates to.  It reads the selected run from
  ``TWPreviousRuns``, loads its ``.omrat`` snapshot + JSON sidecars,
  and feeds them to the matching calc-method on ``self.calc``.

Pure-data work for the various interactive popups lives in:

* :mod:`compute.visualization` -- drift / powered visualisers,
* :mod:`compute.ship_collision_model` -- collision breakdown dialogs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import QEvent, QLocale, QObject, Qt
from qgis.PyQt.QtGui import QCursor, QKeySequence
from qgis.PyQt.QtWidgets import QApplication, QMessageBox, QProgressDialog

from omrat_utils.accident_summary import (
    ACCIDENT_TOTAL_KEYS, DISPLAY_MODE_LABELS, DISPLAY_MODES, RESULT_DISPLAY_SETTING,
    SUMMARY_ROWS, catastrophe_label, format_result, normalize_display_mode,
)
from omrat_utils.result_compare import (
    CHANGE, DASH, NO_CONSEQUENCE_NOTE, RunValues, accident_values,
    build_compare_columns, catastrophe_rows, change_text, compare_hint,
    copy_number, exceedance_values, value_copy_text, value_text,
)
from omrat_utils.run_history_mixin import _qt_enum
from omrat_utils.table_clipboard import to_tsv

if TYPE_CHECKING:
    pass


def _copy_role() -> int:
    """Item data role holding a cell's clipboard text (a plain number)."""
    role = _qt_enum(Qt, 'UserRole', 'ItemDataRole.UserRole')
    return int(getattr(role, 'value', role)) + 1


class _ResultCopyKeys(QObject):
    """Ctrl+C (and Ctrl+Insert) on the result tables.

    QGIS binds Copy to *Copy features* in its main window, and a window
    shortcut fires before the key reaches the focused table, so the table
    claims the key on ``ShortcutOverride`` first (as the Traffic table
    does).
    """

    def __init__(self, copy, parent=None):
        super().__init__(parent)
        self._copy = copy

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if event.type() == QEvent.Type.ShortcutOverride and event.matches(QKeySequence.StandardKey.Copy):
            event.accept()      # deliver as a KeyPress to the table, not to QGIS
            return True
        if event.type() == QEvent.Type.KeyPress and event.matches(QKeySequence.StandardKey.Copy):
            self._copy()
            return True
        return False


class _ViewProgress:
    """Wait cursor + indeterminate QProgressDialog for the View slots.

    The matplotlib build for powered/drifting visualisations runs on
    the UI thread, so the user otherwise sees a frozen window after
    clicking View.  We paint the dialog via processEvents() before the
    work starts so the click is visibly acknowledged.
    """

    def __init__(self, parent, label: str):
        self._parent = parent
        self._label = label
        self._dlg: QProgressDialog | None = None
        self._cursor_set = False

    def __enter__(self):
        try:
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            self._cursor_set = True
            self._dlg = QProgressDialog(
                f"Building {self._label} visualization...",
                None, 0, 0, self._parent,
            )
            self._dlg.setWindowTitle("OMRAT")
            self._dlg.setWindowModality(Qt.WindowModality.WindowModal)
            self._dlg.setMinimumDuration(0)
            self._dlg.setCancelButton(None)
            self._dlg.show()
            QApplication.processEvents()
        except Exception:  # nosec B110 B112
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._dlg is not None:
                self._dlg.close()
                self._dlg = None
        finally:
            if self._cursor_set:
                QApplication.restoreOverrideCursor()
                self._cursor_set = False
        return False


class AccidentResultsMixin:
    """``TWAccidentResults`` setup + per-row View-button dispatcher."""

    # (Accident type label, LEP* widget name on main_widget,
    #  View pushbutton name on main_widget, slot on self).
    # Order is the row order of the table.
    _ACCIDENT_ROWS: tuple[tuple[str, str, str, str], ...] = (
        ('Drifting allision', 'LEPDriftAllision',
         'pbViewDriftingAllision', 'show_drift_allision'),
        ('Drifting grounding', 'LEPDriftingGrounding',
         'pbViewDriftingGrounding', 'show_drift_grounding'),
        ('Powered allision', 'LEPPoweredAllision',
         'pbViewPoweredAllision', 'show_powered_allision'),
        ('Powered grounding', 'LEPPoweredGrounding',
         'pbViewPoweredGrounding', 'show_powered_grounding'),
        ('Overtaking collision', 'LEPOvertakingCollision',
         'pbViewOvertakingCollision', 'show_overtaking_collision'),
        ('Head-on collision', 'LEPHeadOnCollision',
         'pbViewHeadOnCollision', 'show_head_on_collision'),
        ('Crossing collision', 'LEPCrossingCollision',
         'pbViewCrossingCollision', 'show_crossing_collision'),
        ('Merging collision', 'LEPMergingCollision',
         'pbViewMergingCollision', 'show_merging_collision'),
        ('Bend collision', 'LEPBendCollision',
         'pbViewBendCollision', 'show_bend_collision'),
    )

    # View slot -> (calc-method-name, label, optional breakdown key).
    _VIEW_DISPATCH: dict[str, tuple[str, str, str | None]] = {
        'show_drift_allision': (
            'run_drift_visualization', 'Drifting allision', None,
        ),
        'show_drift_grounding': (
            'run_drift_grounding_visualization', 'Drifting grounding', None,
        ),
        'show_powered_allision': (
            'run_powered_allision_visualization', 'Powered allision', None,
        ),
        'show_powered_grounding': (
            'run_powered_grounding_visualization',
            'Powered grounding', None,
        ),
        'show_overtaking_collision': (
            'run_collision_breakdown_dialog',
            'Overtaking collision', 'overtaking',
        ),
        'show_head_on_collision': (
            'run_collision_breakdown_dialog',
            'Head-on collision', 'head_on',
        ),
        'show_crossing_collision': (
            'run_collision_breakdown_dialog',
            'Crossing collision', 'crossing',
        ),
        'show_merging_collision': (
            'run_collision_breakdown_dialog',
            'Merging collision', 'merging',
        ),
        'show_bend_collision': (
            'run_collision_breakdown_dialog',
            'Bend collision', 'bend',
        ),
    }

    # ------------------------------------------------------------------
    # Table setup
    # ------------------------------------------------------------------
    def _setup_accident_results_table(self) -> None:
        """Configure ``TWAccidentResults`` and create the legacy LEP*
        line-edits as hidden Python attributes on ``main_widget``.

        Many compute / test paths still call ``LEPDriftAllision.setText``
        etc., so we keep the same widget names available.  The table does
        not show them (2026-11-09): it shows saved runs only -- the
        baseline and the runs selected in ``TWPreviousRuns``
        (:meth:`_show_run_comparison`).  A finished run is saved and
        selected, so it appears there.
        """
        tw = getattr(self.main_widget, 'TWAccidentResults', None)
        if tw is None:
            return

        self._ensure_legacy_lep_widgets()
        self._configure_accident_table(tw)
        self._populate_accident_rows(tw)
        self._wire_result_table_copy(tw)
        self._setup_compare_hint()
        self._setup_catastrophe_results_table()
        self._setup_result_display_mode_combo()

    # ------------------------------------------------------------------
    # Display mode (frequency per year <-> years between incidents)
    # ------------------------------------------------------------------
    @staticmethod
    def _qsettings():
        # Go through ``run_history_mixin`` so tests that swap its
        # ``QSettings`` for an in-memory stand-in cover this mixin too.
        from omrat_utils import run_history_mixin as _rhm
        return _rhm.QSettings()

    def _result_display_mode(self) -> str:
        """Current display mode, read from QSettings (default frequency)."""
        try:
            value = self._qsettings().value(RESULT_DISPLAY_SETTING, '', type=str)
        except Exception:  # nosec B110 B112
            value = ''
        return normalize_display_mode(value)

    def _set_result_display_mode(self, mode: str) -> None:
        try:
            self._qsettings().setValue(RESULT_DISPLAY_SETTING, normalize_display_mode(mode))
        except Exception:  # nosec B110 B112
            pass

    def _format_result(self, value) -> str:
        """Table text for an annual frequency in the active display mode."""
        return format_result(value, self._result_display_mode())

    def _accident_headers(self) -> list[str]:
        """Headers of the fixed columns; run columns go between them."""
        return ['Accident type', 'View']

    def _catastrophe_headers(self) -> list[str]:
        """Headers of the fixed columns; run columns are appended."""
        return ['Catastrophe level', 'Threshold (m^3)']

    def _setup_result_display_mode_combo(self) -> None:
        """Fill ``cbResultDisplayMode`` and hook it to the tables."""
        cb = getattr(self.main_widget, 'cbResultDisplayMode', None)
        if cb is None:
            return
        mode = self._result_display_mode()
        try:
            cb.blockSignals(True)
            cb.clear()
            for m in DISPLAY_MODES:
                cb.addItem(self.tr(DISPLAY_MODE_LABELS[m]), m)
            cb.setCurrentIndex(DISPLAY_MODES.index(mode))
        finally:
            cb.blockSignals(False)
        try:
            cb.currentIndexChanged.connect(self._on_result_display_mode_changed)
        except Exception:  # nosec B110 B112
            pass
        self._apply_result_display_mode()

    def _on_result_display_mode_changed(self, index: int) -> None:
        cb = getattr(self.main_widget, 'cbResultDisplayMode', None)
        mode = None
        if cb is not None:
            mode = cb.itemData(index)
        if mode is None and 0 <= int(index) < len(DISPLAY_MODES):
            mode = DISPLAY_MODES[int(index)]
        self._set_result_display_mode(normalize_display_mode(mode))
        self._apply_result_display_mode()

    def _apply_result_display_mode(self) -> None:
        """Re-render both result tables in the active display mode.

        The run history always holds annual frequencies; only the cell
        text, the copied numbers and the change (computed in the unit
        shown) follow the mode.  The run columns are rebuilt through the
        previous-runs selection handler.
        """
        lbl = getattr(self.main_widget, 'lblCatastropheResults', None)
        if lbl is not None:
            try:
                lbl.setText(self.tr(catastrophe_label(self._result_display_mode())))
            except Exception:  # nosec B110 B112
                pass
        refresh = getattr(self, '_on_previous_runs_selection_changed', None)
        if callable(refresh):
            try:
                refresh()
                return
            except Exception:  # nosec B110 B112
                pass
        self._show_run_comparison(None, [])

    def _setup_catastrophe_results_table(self) -> None:
        """Configure ``TWCatastropheResults`` -- the small exceedance table
        below ``TWAccidentResults`` on the Run Analysis tab.  Rows and run
        columns come from :meth:`_show_run_comparison`.
        """
        tw = getattr(self.main_widget, 'TWCatastropheResults', None)
        if tw is None:
            return
        tw.setColumnCount(len(self._catastrophe_headers()))
        tw.setHorizontalHeaderLabels(self._catastrophe_headers())
        tw.setRowCount(0)
        self._configure_result_table(tw)
        self._wire_result_table_copy(tw)

    def _configure_result_table(self, tw) -> None:
        """Read-only, cell-wise selection (copy any block to Excel), every
        column sized to its contents.  A stretched label column was
        squeezed to nothing once a few run columns were added."""
        from qgis.PyQt import QtWidgets
        from qgis.PyQt.QtWidgets import QHeaderView
        AIV = QtWidgets.QAbstractItemView
        tw.verticalHeader().setVisible(False)
        tw.setEditTriggers(_qt_enum(
            AIV, 'NoEditTriggers', 'EditTrigger.NoEditTriggers',
        ))
        tw.setSelectionBehavior(_qt_enum(
            AIV, 'SelectItems', 'SelectionBehavior.SelectItems',
        ))
        tw.setSelectionMode(_qt_enum(
            AIV, 'ExtendedSelection', 'SelectionMode.ExtendedSelection',
        ))
        tw.horizontalHeader().setStretchLastSection(False)
        try:
            # The no-index form is also the mode of columns inserted later.
            tw.horizontalHeader().setSectionResizeMode(_qt_enum(
                QHeaderView, 'ResizeToContents', 'ResizeMode.ResizeToContents',
            ))
        except Exception:  # nosec B110 B112
            pass

    def _reset_accident_table_to_base(self) -> None:
        """Strip the run columns from ``TWAccidentResults``: back to
        Accident type | View."""
        tw = getattr(self.main_widget, 'TWAccidentResults', None)
        if tw is None:
            return
        # Run columns sit between the label column and View; drop them
        # right-to-left so the View column (cell widgets) stays put.
        while tw.columnCount() > 2:
            tw.removeColumn(tw.columnCount() - 2)
        tw.setHorizontalHeaderLabels(self._accident_headers())

    def _reset_catastrophe_table_to_base(self) -> None:
        """Strip ``TWCatastropheResults`` to its two empty fixed columns."""
        tw = getattr(self.main_widget, 'TWCatastropheResults', None)
        if tw is None:
            return
        tw.setRowCount(0)
        tw.setColumnCount(len(self._catastrophe_headers()))
        tw.setHorizontalHeaderLabels(self._catastrophe_headers())

    # ------------------------------------------------------------------
    # Run columns: baseline + selected runs
    # ------------------------------------------------------------------
    def _show_run_comparison(self, baseline_run, runs) -> None:
        """Fill both result tables with the baseline run (``RunMeta`` or
        ``None``) and the selected runs (``RunMeta`` list); see
        :mod:`omrat_utils.result_compare` for the layout."""
        from omrat_utils.run_history_mixin import load_consequence_sidecar

        runs = list(runs or [])
        others = [r for r in runs if baseline_run is None or r.run_id != baseline_run.run_id]

        base_values = self._accident_run_values(baseline_run) if baseline_run is not None else None
        self._render_accident_comparison(base_values, [self._accident_run_values(r) for r in others])

        base_result = load_consequence_sidecar(baseline_run) if baseline_run is not None else None
        results = [load_consequence_sidecar(r) for r in others]
        rows = catastrophe_rows([base_result] + results)
        names = [name for name, _qty in rows]
        cat_base = None
        if baseline_run is not None:
            cat_base = self._catastrophe_run_values(baseline_run, base_result, names)
        self._render_catastrophe_comparison(
            rows, cat_base,
            [self._catastrophe_run_values(r, res, names) for r, res in zip(others, results)],
        )
        self._set_compare_hint(compare_hint(
            getattr(baseline_run, 'name', None) if baseline_run is not None else None, len(others),
        ))

    @staticmethod
    def _accident_run_values(run) -> RunValues:
        try:
            totals = run.totals_dict()
        except Exception:  # nosec B110 B112
            totals = {}
        return RunValues(name=str(run.name), values=accident_values(totals), run_id=run.run_id)

    @staticmethod
    def _catastrophe_run_values(run, result, names) -> RunValues:
        values = exceedance_values(result, names)
        return RunValues(
            name=str(run.name), values=values, run_id=run.run_id,
            note=NO_CONSEQUENCE_NOTE if values is None else '',
        )

    def _render_accident_comparison(self, baseline: RunValues | None, runs: list[RunValues]) -> None:
        """Run columns of ``TWAccidentResults``, inserted before View."""
        tw = getattr(self.main_widget, 'TWAccidentResults', None)
        if tw is None:
            return
        self._reset_accident_table_to_base()
        mode = self._result_display_mode()
        n_accidents = len(self._ACCIDENT_ROWS)
        columns = build_compare_columns(baseline, runs, n_accidents + len(SUMMARY_ROWS), mode)
        # A blank cell = a part the run did not calculate (never zero).
        self._insert_run_columns(tw, columns, first_col=1, bold_from=n_accidents, missing='')

    def _render_catastrophe_comparison(self, rows, baseline: RunValues | None, runs: list[RunValues]) -> None:
        """Level rows + run columns of ``TWCatastropheResults``."""
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        tw = getattr(self.main_widget, 'TWCatastropheResults', None)
        if tw is None:
            return
        self._reset_catastrophe_table_to_base()
        tw.setRowCount(len(rows))
        for r, (name, qty) in enumerate(rows):
            tw.setItem(r, 0, QTableWidgetItem(name))
            qty_item = QTableWidgetItem(f'{qty:.2f}' if qty is not None else '')
            qty_item.setData(_copy_role(), copy_number(qty))
            tw.setItem(r, 1, qty_item)
        columns = build_compare_columns(baseline, runs, len(rows), self._result_display_mode())
        self._insert_run_columns(tw, columns, first_col=2, bold_from=None, missing=DASH)

    def _insert_run_columns(self, tw, columns, *, first_col: int, bold_from: int | None,
                            missing: str) -> None:
        """Insert ``columns`` (``CompareColumn``) at ``first_col``; each cell
        keeps its clipboard number under :func:`_copy_role`, with a ``.``
        decimal point (the locale's is applied when copying)."""
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        mode = self._result_display_mode()
        role = _copy_role()
        align = _qt_enum(Qt, 'AlignRight', 'AlignmentFlag.AlignRight') | _qt_enum(
            Qt, 'AlignVCenter', 'AlignmentFlag.AlignVCenter')
        for offset, column in enumerate(columns):
            col = first_col + offset
            tw.insertColumn(col)
            header = QTableWidgetItem(column.header)
            header.setToolTip(column.tooltip)
            tw.setHorizontalHeaderItem(col, header)
            for row, v in enumerate(column.cells):
                if column.kind == CHANGE:
                    text, copy = change_text(v), copy_number(v)
                else:
                    text = value_text(v, mode, has_data=column.has_data, missing=missing)
                    copy = value_copy_text(v, mode) if column.has_data else ''
                item = QTableWidgetItem(text)
                item.setData(role, copy)
                item.setTextAlignment(align)
                if bold_from is not None and row >= bold_from:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                tw.setItem(row, col, item)

    def _setup_compare_hint(self) -> None:
        """An italic, word-wrapped line between the accident table's
        caption and the table saying what the run columns show
        (``lblCompareHint``).  On its own line so a long run name cannot
        push the Show as box out of the window."""
        from qgis.PyQt import QtWidgets
        if getattr(self.main_widget, 'lblCompareHint', None) is not None:
            return
        tw = getattr(self.main_widget, 'TWAccidentResults', None)
        parent = tw.parentWidget() if tw is not None else None
        layout = parent.layout() if parent is not None else None
        index = layout.indexOf(tw) if layout is not None else -1
        if index < 0:
            return
        try:
            lbl = QtWidgets.QLabel(parent)
            lbl.setObjectName('lblCompareHint')
            lbl.setWordWrap(True)
            font = lbl.font()
            font.setItalic(True)
            lbl.setFont(font)
            layout.insertWidget(index, lbl)
            self.main_widget.lblCompareHint = lbl
        except Exception:  # nosec B110 B112
            return
        self._set_compare_hint(compare_hint(None, 0))

    def _set_compare_hint(self, text: str) -> None:
        lbl = getattr(self.main_widget, 'lblCompareHint', None)
        if lbl is not None:
            try:
                lbl.setText(self.tr(text))
            except Exception:  # nosec B110 B112
                pass

    # ------------------------------------------------------------------
    # Copy to Excel
    # ------------------------------------------------------------------
    @staticmethod
    def _decimal_point() -> str:
        try:
            return str(QLocale().decimalPoint()) or '.'
        except Exception:  # nosec B110 B112
            return '.'

    def _wire_result_table_copy(self, tw) -> None:
        """Ctrl+C copies the selected cells; the right-click menu adds
        *Copy table with headers*.  Wired once per table."""
        filters = getattr(self, '_result_copy_filters', None)
        if filters is None:
            filters = self._result_copy_filters = {}
        if tw.objectName() in filters:
            return
        try:
            keys = _ResultCopyKeys(lambda t=tw: self._copy_result_table(t), tw)
            tw.installEventFilter(keys)
            filters[tw.objectName()] = keys
            tw.setContextMenuPolicy(_qt_enum(
                Qt, 'CustomContextMenu', 'ContextMenuPolicy.CustomContextMenu',
            ))
            tw.customContextMenuRequested.connect(
                lambda pos, t=tw: self._result_table_menu(t, pos),
            )
        except Exception:  # nosec B110 B112
            pass

    def _result_table_menu(self, tw, pos) -> None:
        from qgis.PyQt.QtWidgets import QMenu
        menu = QMenu(tw)
        copy = menu.addAction(self.tr('Copy'))
        copy.setEnabled(bool(tw.selectedRanges()))
        copy.triggered.connect(lambda *_: self._copy_result_table(tw))
        menu.addAction(self.tr('Copy table with headers')).triggered.connect(
            lambda *_: self._copy_result_table(tw, whole=True),
        )
        menu.exec(tw.viewport().mapToGlobal(pos))

    def _copy_result_table(self, tw, *, whole: bool = False) -> str:
        """Put the selection (or the whole table with its headers) on the
        clipboard as tab-separated text and return that text."""
        text = self._result_table_tsv(tw, whole=whole)
        if text:
            try:
                QApplication.clipboard().setText(text)
            except Exception:  # nosec B110 B112
                pass
        return text

    def _result_table_tsv(self, tw, *, whole: bool = False) -> str:
        """Tab-separated cells for Excel: numbers as plain numbers with the
        locale's decimal point (each cell's copy role), labels as text.
        The View column (cell widgets) is left out.

        The decimal point is applied here, when copying: the copy role
        holds ``.``.  It was baked in when the table was drawn until
        2026-11-12, so the copy followed the locale of that moment.
        """
        skip = {
            c for c in range(tw.columnCount())
            if any(tw.cellWidget(r, c) is not None for r in range(tw.rowCount()))
        }
        if whole:
            rows = list(range(tw.rowCount()))
            cols = [c for c in range(tw.columnCount()) if c not in skip]
        else:
            ranges = tw.selectedRanges()
            rows = sorted({r for rng in ranges for r in range(rng.topRow(), rng.bottomRow() + 1)})
            cols = sorted({
                c for rng in ranges for c in range(rng.leftColumn(), rng.rightColumn() + 1)
            } - skip)
        if not cols or not (rows or whole):
            return ''
        role = _copy_role()
        decimal_point = self._decimal_point()
        lines: list[list[str]] = []
        if whole:
            lines.append([
                tw.horizontalHeaderItem(c).text() if tw.horizontalHeaderItem(c) is not None else ''
                for c in cols
            ])
        for r in rows:
            cells: list[str] = []
            for c in cols:
                item = tw.item(r, c)
                if item is None:
                    cells.append('')
                    continue
                copy = item.data(role)
                if copy is None:
                    cells.append(item.text())
                else:
                    cells.append(str(copy).replace('.', decimal_point))
            lines.append(cells)
        return to_tsv(lines)

    def _ensure_legacy_lep_widgets(self) -> None:
        """Create the ``LEP*`` / ``pbView*`` widgets as hidden children
        of ``main_widget`` if they don't already exist.

        They used to live in the .ui file; we keep them programmatically
        so external callers (compute models, tests) that do
        ``LEPDriftAllision.setText`` keep working.
        """
        from qgis.PyQt import QtWidgets
        for _label, le_name, pb_name, _slot_name in self._ACCIDENT_ROWS:
            le = getattr(self.main_widget, le_name, None)
            if le is None:
                le = QtWidgets.QLineEdit(self.main_widget)
                le.setVisible(False)
                setattr(self.main_widget, le_name, le)
            pb = getattr(self.main_widget, pb_name, None)
            if pb is None:
                pb = QtWidgets.QPushButton(self.main_widget)
                pb.setVisible(False)
                setattr(self.main_widget, pb_name, pb)

    def _configure_accident_table(self, tw) -> None:
        headers = self._accident_headers()
        tw.setColumnCount(len(headers))
        tw.setHorizontalHeaderLabels(headers)
        tw.setRowCount(
            len(AccidentResultsMixin._ACCIDENT_ROWS) + len(SUMMARY_ROWS),
        )
        self._configure_result_table(tw)

    def _populate_accident_rows(self, tw) -> None:
        """Row labels (nine accidents + three bold summary rows) and the
        View buttons in the last column; the run columns go in between."""
        from qgis.PyQt import QtWidgets
        view_col = tw.columnCount() - 1
        for row, (label, _le_name, _pb_name, slot_name) in enumerate(
            self._ACCIDENT_ROWS,
        ):
            tw.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
            btn = QtWidgets.QPushButton(self.tr('View'))
            btn.setToolTip(self.tr('Open the details of the run selected in Previous runs'))
            slot = getattr(self, slot_name, None)
            if callable(slot):
                btn.clicked.connect(slot)
            else:
                btn.setEnabled(False)
            tw.setCellWidget(row, view_col, btn)

        n_accidents = len(self._ACCIDENT_ROWS)
        for offset, (label, _keys) in enumerate(SUMMARY_ROWS):
            item = QtWidgets.QTableWidgetItem(label)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            tw.setItem(n_accidents + offset, 0, item)

    def _clear_accident_results(self, keys) -> None:
        """Blank the rows of ``keys`` (``ACCIDENT_TOTAL_KEYS`` names): a
        model part the last run did not calculate has no value."""
        for row, key in enumerate(ACCIDENT_TOTAL_KEYS):
            if key not in keys:
                continue
            le = getattr(self.main_widget, self._ACCIDENT_ROWS[row][1], None)
            if le is not None:
                le.setText('')

    # ------------------------------------------------------------------
    # View dispatch
    # ------------------------------------------------------------------
    def _require_single_selected_run(self):
        """Return the single selected ``RunMeta`` or surface a popup.

        Returns ``None`` (after showing a popup) if 0 or >1 rows are
        selected in ``TWPreviousRuns``.
        """
        try:
            run_ids = self._selected_run_ids()
        except Exception:  # nosec B110 B112
            run_ids = []
        if len(run_ids) == 0:
            QMessageBox.information(
                self.main_widget,
                self.tr('Select a run'),
                self.tr(
                    "Pick a run in the Previous-runs table at the top of "
                    "the Run Analysis tab before clicking View."
                ),
            )
            return None
        if len(run_ids) > 1:
            QMessageBox.information(
                self.main_widget,
                self.tr('Select only one run'),
                self.tr(
                    "View shows the breakdown of a single run.  Select "
                    "exactly one row in the Previous-runs table."
                ),
            )
            return None
        try:
            from omrat_utils.run_history import RunHistory
            runs = RunHistory().compare_runs(run_ids)
        except Exception as exc:
            self.show_error_popup(str(exc), '_require_single_selected_run')
            return None
        return runs[0] if runs else None

    def _load_run_inputs_and_collision_report(
        self, run,
    ) -> tuple[dict | None, dict | None, dict | None]:
        """Load ``data`` (.omrat), collision_report, drifting_report
        from disk for ``run``.  Any may be ``None`` when the file
        isn't present (older runs that pre-date the sidecar code).
        """
        data = None
        collision_report = None
        drifting_report = None
        try:
            gpkg_path = (
                run.gpkg_path() if hasattr(run, 'gpkg_path') else None
            )
        except Exception:  # nosec B110 B112
            gpkg_path = None
        if gpkg_path is None:
            return data, collision_report, drifting_report
        stem = Path(gpkg_path).with_suffix('')
        for path, label, target in (
            (stem.with_suffix('.omrat'), 'data', 'data'),
            (Path(str(stem) + '.collision.json'), 'collision', 'cr'),
            (Path(str(stem) + '.drifting.json'), 'drifting', 'dr'),
        ):
            if not path.is_file():
                continue
            try:
                with path.open('r', encoding='utf-8') as f:
                    payload = json.load(f)
            except Exception as exc:
                QgsMessageLog.logMessage(
                    f"Could not read {path}: {exc}",
                    'OMRAT', Qgis.MessageLevel.Warning,
                )
                continue
            if target == 'data':
                data = payload
            elif target == 'cr':
                collision_report = payload
            elif target == 'dr':
                drifting_report = payload
        return data, collision_report, drifting_report

    def _dispatch_view(self, slot_name: str) -> None:
        spec = self._VIEW_DISPATCH.get(slot_name)
        if spec is None:
            return
        method_name, label, breakdown_key = spec

        run = self._require_single_selected_run()
        if run is None:
            return
        if self._not_calculated_in_run(slot_name, run, label):
            return
        data, collision_report, drifting_report = (
            self._load_run_inputs_and_collision_report(run)
        )
        if data is None:
            QMessageBox.information(
                self.main_widget,
                self.tr('Snapshot missing'),
                self.tr(
                    "No .omrat snapshot was found next to this run's "
                    "GeoPackage, so the interactive visualiser cannot "
                    "rebuild its inputs.  Re-run the model to produce "
                    "a snapshot, or pick a newer run."
                ),
            )
            return
        if self.calc is None:
            return
        method = getattr(self.calc, method_name, None)
        if not callable(method):
            return
        if method_name == 'run_collision_breakdown_dialog':
            # The breakdown is a plain table that opens instantly and
            # runs a modal exec() loop.  Wrapping it in the progress
            # dialog would leave "Building ... visualization" on screen
            # for as long as the table is open, so no progress here.
            self._invoke_collision_breakdown(
                method, breakdown_key, collision_report,
            )
            return
        # All early-return paths cleared: actual build starts here.  Show
        # a wait cursor + indeterminate progress so the user gets
        # immediate feedback that the click registered.  The
        # visualisation still runs on the UI thread; processEvents
        # paints the dialog before the freeze.
        with self._view_progress(label):
            self._invoke_drift_or_powered_visualiser(
                method, data, drifting_report,
                powered_reports=self._load_powered_reports(run),
            )

    def _not_calculated_in_run(self, slot_name: str, run, label: str) -> bool:
        """Tell the user (and return True) when ``run`` skipped this
        accident type in the Calculation matrix -- its history total is
        empty, while every calculated type has a number (0 included)."""
        row = next((i for i, r in enumerate(self._ACCIDENT_ROWS) if r[3] == slot_name), None)
        if row is None:
            return False
        try:
            value = run.totals_dict().get(ACCIDENT_TOTAL_KEYS[row])
        except Exception:  # nosec B110 B112
            return False
        if value is not None:
            return False
        QMessageBox.information(
            self.main_widget,
            self.tr('Not calculated'),
            self.tr(
                "{label} was not calculated in the run '{name}' (it was "
                "left out in the Calculation matrix).  Pick a run that "
                "includes it."
            ).format(label=label, name=getattr(run, 'name', '')),
        )
        return True

    def _view_progress(self, label: str):
        return _ViewProgress(self.main_widget, label)

    @staticmethod
    def _load_powered_reports(run) -> dict | None:
        """The run's ``<stem>.powered.json`` (per-leg powered shares), or ``None``."""
        try:
            from omrat_utils.run_history_mixin import load_powered_sidecar
            return load_powered_sidecar(run)
        except Exception:  # nosec B110 B112
            return None

    def _invoke_collision_breakdown(
        self, method, breakdown_key, collision_report,
    ) -> None:
        previous = getattr(self.calc, 'collision_report', None)
        try:
            self.calc.collision_report = collision_report or {}
            method(breakdown_key)
        finally:
            self.calc.collision_report = previous

    def _invoke_drift_or_powered_visualiser(
        self, method, data, drifting_report, powered_reports=None,
    ) -> None:
        prev_dr = getattr(self.calc, 'drifting_report', None)
        prev_pr = getattr(self.calc, 'powered_view_reports', None)
        try:
            if drifting_report is not None:
                self.calc.drifting_report = drifting_report
            # Always set (also to None): a previous View's run must not leak in.
            self.calc.powered_view_reports = powered_reports
            method(data)
        finally:
            self.calc.drifting_report = prev_dr
            self.calc.powered_view_reports = prev_pr

    # ------------------------------------------------------------------
    # Public View slots (one per accident type)
    # ------------------------------------------------------------------
    def show_drift_allision(self):
        self._dispatch_view('show_drift_allision')

    def show_drift_grounding(self):
        self._dispatch_view('show_drift_grounding')

    def show_powered_allision(self):
        self._dispatch_view('show_powered_allision')

    def show_powered_grounding(self):
        self._dispatch_view('show_powered_grounding')

    def show_overtaking_collision(self):
        self._dispatch_view('show_overtaking_collision')

    def show_head_on_collision(self):
        self._dispatch_view('show_head_on_collision')

    def show_crossing_collision(self):
        self._dispatch_view('show_crossing_collision')

    def show_merging_collision(self):
        self._dispatch_view('show_merging_collision')

    def show_bend_collision(self):
        self._dispatch_view('show_bend_collision')
