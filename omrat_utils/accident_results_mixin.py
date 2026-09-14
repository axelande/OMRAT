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
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QCursor
from qgis.PyQt.QtWidgets import QApplication, QMessageBox, QProgressDialog

from omrat_utils.accident_summary import (
    ACCIDENT_TOTAL_KEYS, DISPLAY_MODE_LABELS, DISPLAY_MODES, RESULT_DISPLAY_SETTING,
    SUMMARY_ROWS, catastrophe_header, catastrophe_label, format_result,
    normalize_display_mode, parse_probability, result_header,
    summary_values,
)
from omrat_utils.run_history_mixin import _qt_enum

if TYPE_CHECKING:
    pass

# Placeholder for cells with no value (matches _format_delta_pct).
_DASH = '\u2014'


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
        etc., so we keep the same widget names available.  Edits are
        forwarded to the new table cell via ``_on_lep_text_changed``.
        """
        tw = getattr(self.main_widget, 'TWAccidentResults', None)
        if tw is None:
            return

        self._ensure_legacy_lep_widgets()
        self._configure_accident_table(tw)
        self._populate_accident_rows(tw)
        self._wire_clipboard_copy_shortcut(tw)
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
        return ['Accident type', result_header(self._result_display_mode()), 'View']

    def _catastrophe_headers(self) -> list[str]:
        return ['Catastrophe level', 'Threshold (m^3)', catastrophe_header(self._result_display_mode())]

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

        The stored values (hidden ``LEP*`` widgets, the live consequence
        result and the run history) are always annual frequencies; only
        the cell text changes.  Comparison columns are rebuilt through
        the previous-runs selection handler so they follow the mode too.
        """
        from qgis.PyQt import QtWidgets
        tw = getattr(self.main_widget, 'TWAccidentResults', None)
        if tw is not None:
            for row, (_label, le_name, _pb, _slot) in enumerate(self._ACCIDENT_ROWS):
                le = getattr(self.main_widget, le_name, None)
                text = le.text() if le is not None else ''
                tw.setItem(row, 1, QtWidgets.QTableWidgetItem(
                    self._format_result(parse_probability(text)),
                ))
            self._refresh_summary_rows(tw)
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
        self._reset_accident_table_to_base()
        self._reset_catastrophe_table_to_base()

    def _setup_catastrophe_results_table(self) -> None:
        """Configure ``TWCatastropheResults`` -- the small annual-frequency
        table that sits below ``TWAccidentResults`` on the Run Analysis
        tab.  Headers are set here; rows get (re)populated each run from
        ``_populate_catastrophe_results_table``.
        """
        from qgis.PyQt import QtWidgets
        from qgis.PyQt.QtWidgets import QHeaderView

        AIV = QtWidgets.QAbstractItemView
        tw = getattr(self.main_widget, 'TWCatastropheResults', None)
        if tw is None:
            return
        tw.setColumnCount(3)
        tw.setHorizontalHeaderLabels(self._catastrophe_headers())
        tw.setRowCount(0)
        tw.verticalHeader().setVisible(False)
        tw.setEditTriggers(_qt_enum(
            AIV, 'NoEditTriggers', 'EditTrigger.NoEditTriggers',
        ))
        tw.setSelectionBehavior(_qt_enum(
            AIV, 'SelectRows', 'SelectionBehavior.SelectRows',
        ))
        try:
            mode_stretch = _qt_enum(
                QHeaderView, 'Stretch', 'ResizeMode.Stretch',
            )
            mode_resize = _qt_enum(
                QHeaderView, 'ResizeToContents', 'ResizeMode.ResizeToContents',
            )
            tw.horizontalHeader().setSectionResizeMode(0, mode_stretch)
            tw.horizontalHeader().setSectionResizeMode(1, mode_resize)
            tw.horizontalHeader().setSectionResizeMode(2, mode_resize)
        except Exception:  # nosec B110 B112
            pass
        self._wire_clipboard_copy_shortcut(tw)

    def _reset_accident_table_to_base(self) -> None:
        """Strip per-run comparison columns from ``TWAccidentResults``.

        ``_fill_result_fields_from_runs`` (in :class:`RunHistoryMixin`)
        inserts two extra columns per selected run before the View column.
        Selecting nothing or selecting different runs has to put the table
        back to its 3-column base layout (Accident type / Probability /
        View) before a fresh fill -- this method does that.
        """
        tw = getattr(self.main_widget, 'TWAccidentResults', None)
        if tw is None:
            return
        # Base layout has exactly three columns; anything past that came
        # from a previous-runs comparison fill.  Drop them right-to-left
        # so column indices stay stable.
        while tw.columnCount() > 3:
            tw.removeColumn(tw.columnCount() - 2)
        # Reset headers + the View column header to the canonical labels
        # so the comparison-fill labels don't bleed across selections.
        tw.setHorizontalHeaderLabels(self._accident_headers())

    def _populate_catastrophe_results_table(self, consequence_result) -> None:
        """Populate ``TWCatastropheResults`` from a ``consequence_result``
        dict produced by :func:`compute.consequence.compute_catastrophe_exceedance`.

        The result is remembered as the *live* (current model) values so
        the table can be restored after a previous-runs comparison.
        Rows are written in the same (ascending volume) order returned by
        the calculation.  ``None`` or missing levels clear the table.
        """
        self._live_consequence_result = (
            consequence_result if isinstance(consequence_result, dict) else None
        )
        self._reset_catastrophe_table_to_base()

    @staticmethod
    def _catastrophe_levels(result) -> list[dict]:
        """``levels`` list of a consequence-result dict (``[]`` if absent)."""
        if not isinstance(result, dict):
            return []
        return [lvl for lvl in (result.get('levels', []) or []) if isinstance(lvl, dict)]

    def _reset_catastrophe_table_to_base(self) -> None:
        """Put ``TWCatastropheResults`` back to its three-column layout
        showing the live (current model) exceedance values.

        ``_fill_catastrophe_table_from_runs`` appends two columns per
        selected previous run and may add level rows the live result
        doesn't have; both are undone here.
        """
        from qgis.PyQt import QtWidgets

        tw = getattr(self.main_widget, 'TWCatastropheResults', None)
        if tw is None:
            return
        while tw.columnCount() > 3:
            tw.removeColumn(tw.columnCount() - 1)
        tw.setHorizontalHeaderLabels(self._catastrophe_headers())
        levels = self._catastrophe_levels(
            getattr(self, '_live_consequence_result', None),
        )
        tw.setRowCount(len(levels))
        for r, lvl in enumerate(levels):
            try:
                name = str(lvl.get('name', ''))
                qty = float(lvl.get('quantity', 0.0))
                exceed = float(lvl.get('exceedance', 0.0))
            except Exception:  # nosec B110 B112
                continue
            tw.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            tw.setItem(r, 1, QtWidgets.QTableWidgetItem(f'{qty:.2f}'))
            tw.setItem(r, 2, QtWidgets.QTableWidgetItem(self._format_result(exceed)))

    def _fill_catastrophe_table_from_runs(self, runs) -> None:
        """Add one exceedance + one delta column per selected previous run
        to ``TWCatastropheResults`` (mirrors ``_fill_result_fields_from_runs``
        for the accident table).

        Values come from the ``.consequence.json`` sidecar written next to
        each run's GeoPackage.  Runs recorded before that sidecar existed
        show a dash in every cell.  Levels are matched by name; a level the
        live result lacks gets its own row.
        """
        from qgis.PyQt import QtWidgets
        from omrat_utils.run_history_mixin import load_consequence_sidecar

        tw = getattr(self.main_widget, 'TWCatastropheResults', None)
        if tw is None:
            return
        self._reset_catastrophe_table_to_base()
        if not runs:
            return

        per_run = [(run, load_consequence_sidecar(run)) for run in runs]

        # name -> row, seeded from the live rows; extra levels appended.
        rows: dict[str, int] = {}
        for r in range(tw.rowCount()):
            item = tw.item(r, 0)
            if item is not None:
                rows[item.text()] = r
        for _run, result in per_run:
            for lvl in self._catastrophe_levels(result):
                name = str(lvl.get('name', ''))
                if name in rows:
                    continue
                r = tw.rowCount()
                tw.insertRow(r)
                tw.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
                try:
                    qty_text = f"{float(lvl.get('quantity', 0.0)):.2f}"
                except Exception:  # nosec B110 B112
                    qty_text = ''
                tw.setItem(r, 1, QtWidgets.QTableWidgetItem(qty_text))
                tw.setItem(r, 2, QtWidgets.QTableWidgetItem(_DASH))
                rows[name] = r

        baseline, delta_header = self._catastrophe_baseline(per_run)

        for run_idx, (run, result) in enumerate(per_run):
            run_label = getattr(run, 'name', None) or f'Run {run_idx + 1}'
            col = tw.columnCount()
            tw.insertColumn(col)
            tw.insertColumn(col + 1)
            header = QtWidgets.QTableWidgetItem(str(run_label))
            if result is None:
                header.setToolTip(
                    'No catastrophe data was saved for this run '
                    '(recorded before the .consequence.json sidecar existed). '
                    'Re-run the model to record it.'
                )
            tw.setHorizontalHeaderItem(col, header)
            tw.setHorizontalHeaderItem(
                col + 1, QtWidgets.QTableWidgetItem(delta_header),
            )
            values = self._exceedance_by_name(result)
            for name, r in rows.items():
                v = values.get(name)
                tw.setItem(
                    r, col,
                    QtWidgets.QTableWidgetItem(self._format_result(v) if v is not None else _DASH),
                )
                tw.setItem(
                    r, col + 1,
                    QtWidgets.QTableWidgetItem(
                        self._format_delta_pct(v, baseline.get(name)),
                    ),
                )

    @classmethod
    def _exceedance_by_name(cls, result) -> dict[str, float]:
        out: dict[str, float] = {}
        for lvl in cls._catastrophe_levels(result):
            try:
                out[str(lvl.get('name', ''))] = float(lvl.get('exceedance', 0.0))
            except Exception:  # nosec B110 B112
                continue
        return out

    def _catastrophe_baseline(self, per_run) -> tuple[dict[str, float], str]:
        """Baseline exceedance per level name for the delta columns.

        Same priority as the accident table: the main run's sidecar, then
        the live result, then the first selected run.
        """
        from omrat_utils.run_history_mixin import load_consequence_sidecar

        main_run = self._main_run_meta()
        if main_run is not None:
            return (
                self._exceedance_by_name(load_consequence_sidecar(main_run)),
                f'\u0394 vs main ({main_run.name}) %',
            )
        live = self._exceedance_by_name(
            getattr(self, '_live_consequence_result', None),
        )
        if live:
            return live, '\u0394 vs current %'
        first_run, first_result = per_run[0]
        return (
            self._exceedance_by_name(first_result),
            f'\u0394 vs {getattr(first_run, "name", "run 1")} %',
        )

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
        from qgis.PyQt import QtWidgets
        from qgis.PyQt.QtWidgets import QHeaderView
        AIV = QtWidgets.QAbstractItemView

        headers = self._accident_headers()
        tw.setColumnCount(len(headers))
        tw.setHorizontalHeaderLabels(headers)
        tw.setRowCount(
            len(AccidentResultsMixin._ACCIDENT_ROWS) + len(SUMMARY_ROWS),
        )
        tw.verticalHeader().setVisible(False)
        tw.setEditTriggers(_qt_enum(
            AIV, 'NoEditTriggers', 'EditTrigger.NoEditTriggers',
        ))
        tw.setSelectionBehavior(_qt_enum(
            AIV, 'SelectRows', 'SelectionBehavior.SelectRows',
        ))
        tw.setSelectionMode(_qt_enum(
            AIV, 'ExtendedSelection', 'SelectionMode.ExtendedSelection',
        ))
        tw.horizontalHeader().setStretchLastSection(False)
        try:
            mode_stretch = _qt_enum(
                QHeaderView, 'Stretch', 'ResizeMode.Stretch',
            )
            mode_resize = _qt_enum(
                QHeaderView,
                'ResizeToContents', 'ResizeMode.ResizeToContents',
            )
            tw.horizontalHeader().setSectionResizeMode(0, mode_stretch)
            tw.horizontalHeader().setSectionResizeMode(1, mode_resize)
            tw.horizontalHeader().setSectionResizeMode(2, mode_resize)
        except Exception:  # nosec B110 B112
            pass

    def _populate_accident_rows(self, tw) -> None:
        from qgis.PyQt import QtWidgets
        for row, (label, le_name, _pb_name, slot_name) in enumerate(
            self._ACCIDENT_ROWS,
        ):
            tw.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
            le = getattr(self.main_widget, le_name, None)
            text = le.text() if le is not None else ''
            tw.setItem(row, 1, QtWidgets.QTableWidgetItem(
                self._format_result(parse_probability(text)),
            ))

            btn = QtWidgets.QPushButton(self.tr('View'))
            slot = getattr(self, slot_name, None)
            if callable(slot):
                btn.clicked.connect(slot)
            else:
                btn.setEnabled(False)
            tw.setCellWidget(row, 2, btn)

            if le is not None:
                try:
                    le.textChanged.connect(
                        lambda txt, r=row: self._on_lep_text_changed(r, txt),
                    )
                except Exception:  # nosec B110 B112
                    pass

        n_accidents = len(self._ACCIDENT_ROWS)
        for offset, (label, _keys) in enumerate(SUMMARY_ROWS):
            row = n_accidents + offset
            item = QtWidgets.QTableWidgetItem(label)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            tw.setItem(row, 0, item)
            tw.setItem(row, 1, QtWidgets.QTableWidgetItem(''))
            tw.setItem(row, 2, QtWidgets.QTableWidgetItem(''))
        self._refresh_summary_rows(tw)

    def _current_accident_totals(self, tw) -> dict[str, float | None]:
        """Annual frequencies of the nine accident rows, keyed like
        ``RunHistory`` totals (``drift_allision`` ...).

        Read from the hidden ``LEP*`` widgets, which always hold the
        frequency; the visible cell may show years between incidents.
        """
        totals: dict[str, float | None] = {}
        for row, key in enumerate(ACCIDENT_TOTAL_KEYS):
            le = getattr(self.main_widget, self._ACCIDENT_ROWS[row][1], None)
            if le is not None:
                totals[key] = parse_probability(le.text())
                continue
            item = tw.item(row, 1) if tw is not None else None
            totals[key] = parse_probability(item.text() if item is not None else None)
        return totals

    def _refresh_summary_rows(self, tw=None) -> None:
        """Recompute the All grounding / All allision / All collisions
        rows from the nine accident rows above them."""
        from qgis.PyQt import QtWidgets
        if tw is None:
            tw = getattr(self.main_widget, 'TWAccidentResults', None)
        if tw is None:
            return
        values = summary_values(self._current_accident_totals(tw))
        n_accidents = len(self._ACCIDENT_ROWS)
        for offset, value in enumerate(values):
            row = n_accidents + offset
            if row >= tw.rowCount():
                break
            item = QtWidgets.QTableWidgetItem(self._format_result(value))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            tw.setItem(row, 1, item)

    def _wire_clipboard_copy_shortcut(self, tw) -> None:
        try:
            from qgis.PyQt.QtGui import QKeySequence, QShortcut
            for ks in (
                QKeySequence.StandardKey.Copy,
                QKeySequence('Ctrl+Insert'),
            ):
                sc = QShortcut(ks, tw)
                sc.activated.connect(
                    lambda t=tw: self._copy_table_selection_to_clipboard(t),
                )
        except Exception:  # nosec B110 B112
            pass

    @staticmethod
    def _copy_table_selection_to_clipboard(tw) -> None:
        """Copy the currently-selected cells of ``tw`` to the clipboard
        as tab-separated values.

        Whole-row selection yields one TSV line per selected row;
        individual cell selection yields the bounding rectangle with
        empty cells where there's no selection.  Cell widgets (the View
        buttons in column 2) are emitted as empty strings.
        """
        try:
            from qgis.PyQt.QtWidgets import QApplication
        except Exception:  # nosec B110 B112
            return
        ranges = tw.selectedRanges()
        if not ranges:
            return
        rows = sorted({
            r for rng in ranges
            for r in range(rng.topRow(), rng.bottomRow() + 1)
        })
        cols = sorted({
            c for rng in ranges
            for c in range(rng.leftColumn(), rng.rightColumn() + 1)
        })
        if not rows or not cols:
            return
        lines: list[str] = []
        for r in rows:
            cells: list[str] = []
            for c in cols:
                if tw.cellWidget(r, c) is not None:
                    cells.append('')
                    continue
                item = tw.item(r, c)
                cells.append(item.text() if item is not None else '')
            lines.append('\t'.join(cells))
        QApplication.clipboard().setText('\n'.join(lines))

    def _on_lep_text_changed(self, row: int, text: str) -> None:
        """Forward edits on a hidden LEP* into the new accident table."""
        try:
            from qgis.PyQt.QtWidgets import QTableWidgetItem
            tw = self.main_widget.TWAccidentResults
            tw.setItem(row, 1, QTableWidgetItem(self._format_result(parse_probability(text))))
            self._refresh_summary_rows(tw)
        except Exception:  # nosec B110 B112
            pass

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
            )

    def _view_progress(self, label: str):
        return _ViewProgress(self.main_widget, label)

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
        self, method, data, drifting_report,
    ) -> None:
        prev_dr = getattr(self.calc, 'drifting_report', None)
        try:
            if drifting_report is not None:
                self.calc.drifting_report = drifting_report
            method(data)
        finally:
            self.calc.drifting_report = prev_dr

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
